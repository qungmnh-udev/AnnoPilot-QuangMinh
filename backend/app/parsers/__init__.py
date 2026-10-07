from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

TASKS = {'BBOX_2D': 'BBOX_2D', 'POLYGON': 'SEGMENTATION', 'MASK': 'SEGMENTATION', 'KEYPOINT': 'KEYPOINT', 'CUBOID_3D': 'CUBOID_3D'}

def task_type(annotations):
    types = {TASKS[a['shape_type']] for a in annotations}
    return next(iter(types)) if len(types) == 1 else ('MIXED' if types else 'UNKNOWN')

def valid_geometry(kind, g):
    def finite(values):
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    if kind == 'BBOX_2D':
        return finite(g.values()) and g['x2'] > g['x1'] and g['y2'] > g['y1']
    if kind == 'POLYGON':
        return len(g['points']) >= 3 and finite([v for p in g['points'] for v in p]) and abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(g['points'], g['points'][1:]+g['points'][:1]))) > 0
    if kind == 'KEYPOINT':
        return len(g['points']) > 0 and finite([p[k] for p in g['points'] for k in ('x','y')])
    if kind == 'CUBOID_3D':
        return finite([v for key in ('position','dimensions','rotation') for v in g[key].values()]) and all(v > 0 for v in g['dimensions'].values())
    return kind == 'MASK' and bool(g.get('rle'))

class BaseAnnotationParser(ABC):
    def __init__(self):
        self.warnings = []

    def add(self, sample, label, kind, geometry, attributes=None, occluded=None, metadata=None):
        if not valid_geometry(kind, geometry):
            self.warnings.append(f"{sample['file_name']}: skipped invalid {kind} geometry")
            return
        sample['annotations'].append(dict(label=str(label), shape_type=kind, geometry=geometry, attributes=attributes or {}, occluded=occluded, source_metadata=metadata or {}))

    @abstractmethod
    def parse(self, paths, media):
        pass

def sample(name, width=None, height=None):
    if not isinstance(name,str) or not name.strip():
        raise ValueError('Annotation sample names must be nonempty strings')
    return dict(file_name=name, width=int(width) if width and int(width)>0 else None, height=int(height) if height and int(height)>0 else None, annotations=[])

def reject_json_constant(value):
    raise ValueError('Non-finite JSON number: '+value)

class CVATParser(BaseAnnotationParser):
    def __init__(self):
        super().__init__()
        self.cvat_task_id = None
        self.cvat_job_id = None

    def parse(self, paths, media):
        import re
        results = []
        for path in paths:
            root = ET.parse(path).getroot()
            if root.tag != 'annotations':
                raise ValueError('Expected CVAT annotations root')

            meta = root.find('meta')
            orig_w, orig_h = None, None
            if meta is not None:
                task = meta.find('task')
                if task is not None:
                    try:
                        t_id = task.findtext('id')
                        if t_id and t_id.strip().isdigit():
                            self.cvat_task_id = int(t_id.strip())
                    except (ValueError, TypeError):
                        pass
                segment = meta.find('.//segment')
                if segment is not None:
                    try:
                        s_id = segment.findtext('id')
                        if s_id and s_id.strip().isdigit():
                            self.cvat_job_id = int(s_id.strip())
                    except (ValueError, TypeError):
                        pass
                orig_size = meta.find('.//original_size')
                if orig_size is not None:
                    try:
                        orig_w = int(orig_size.findtext('width') or 0) or None
                        orig_h = int(orig_size.findtext('height') or 0) or None
                    except (ValueError, TypeError):
                        pass

            for image in root.findall('image'):
                s = sample(image.get('name', str(image.get('id'))), int(image.get('width', 0)) or orig_w, int(image.get('height', 0)) or orig_h)
                for shape in image:
                    try:
                        attrs = {a.get('name'): a.text for a in shape.findall('attribute')}
                        occluded = shape.get('occluded') == '1' if shape.get('occluded') is not None else None
                        if shape.tag == 'box':
                            kind, g = 'BBOX_2D', dict(zip(('x1','y1','x2','y2'), [float(shape.get(k)) for k in ('xtl','ytl','xbr','ybr')]))
                        elif shape.tag in ('polygon','points'):
                            points = [[float(v) for v in p.split(',')] for p in shape.get('points','').split(';')]
                            if any(len(p) != 2 for p in points):
                                raise ValueError('Expected coordinate pairs')
                            kind = 'POLYGON' if shape.tag == 'polygon' else 'KEYPOINT'
                            g = {'points': points if kind == 'POLYGON' else [dict(x=p[0], y=p[1]) for p in points]}
                        elif shape.tag == 'tag':
                            self.warnings.append(f"{s['file_name']}: classification tags are not scored")
                            continue
                        else:
                            self.warnings.append(f"{s['file_name']}: unsupported CVAT shape {shape.tag}")
                            continue
                        self.add(s, shape.get('label','unlabeled'), kind, g, attrs, occluded, dict(shape.attrib))
                    except (ValueError, TypeError, KeyError):
                        self.warnings.append(f"{s['file_name']}: skipped malformed {shape.tag}")
                results.append(s)

            tracks = root.findall('track')
            if tracks:
                media_by_frame = {}
                for m in (media or []):
                    digits = re.findall(r'\d+', m.stem)
                    if digits:
                        try:
                            media_by_frame[int(digits[-1])] = m
                        except ValueError:
                            pass

                total_frames = None
                task = root.find('.//task')
                if task is not None:
                    if task.findtext('size'):
                        try:
                            total_frames = int(task.findtext('size'))
                        except ValueError:
                            pass
                    elif task.findtext('stop_frame'):
                        try:
                            total_frames = int(task.findtext('stop_frame')) + 1
                        except ValueError:
                            pass

                frames_dict = {}
                for track in tracks:
                    track_label = track.get('label', 'unlabeled')
                    track_id = track.get('id')
                    track_attrs = {a.get('name'): a.text for a in track.findall('attribute')}
                    for shape in track:
                        if shape.tag == 'box':
                            if shape.get('outside') == '1':
                                continue
                            try:
                                frame_idx = int(shape.get('frame', '0'))
                                if frame_idx not in frames_dict:
                                    fname = media_by_frame[frame_idx].name if frame_idx in media_by_frame else f"frame_{frame_idx:06d}.PNG"
                                    frames_dict[frame_idx] = sample(fname, orig_w, orig_h)
                                s = frames_dict[frame_idx]
                                g = dict(zip(('x1','y1','x2','y2'), [float(shape.get(k)) for k in ('xtl','ytl','xbr','ybr')]))
                                attrs = dict(track_attrs)
                                attrs.update({a.get('name'): a.text for a in shape.findall('attribute')})
                                if track_id is not None:
                                    attrs['track_id'] = track_id
                                occluded = shape.get('occluded') == '1' if shape.get('occluded') is not None else None
                                meta_attrib = dict(shape.attrib)
                                meta_attrib['track_id'] = track_id
                                self.add(s, track_label, 'BBOX_2D', g, attrs, occluded, meta_attrib)
                            except (ValueError, TypeError, KeyError):
                                self.warnings.append(f"frame {shape.get('frame')}: skipped malformed box in track {track_id}")
                        elif shape.tag in ('polygon', 'points'):
                            if shape.get('outside') == '1':
                                continue
                            try:
                                frame_idx = int(shape.get('frame', '0'))
                                if frame_idx not in frames_dict:
                                    fname = media_by_frame[frame_idx].name if frame_idx in media_by_frame else f"frame_{frame_idx:06d}.PNG"
                                    frames_dict[frame_idx] = sample(fname, orig_w, orig_h)
                                s = frames_dict[frame_idx]
                                points = [[float(v) for v in p.split(',')] for p in shape.get('points','').split(';')]
                                if any(len(p) != 2 for p in points):
                                    raise ValueError('Expected coordinate pairs')
                                kind = 'POLYGON' if shape.tag == 'polygon' else 'KEYPOINT'
                                g = {'points': points if kind == 'POLYGON' else [dict(x=p[0], y=p[1]) for p in points]}
                                attrs = dict(track_attrs)
                                attrs.update({a.get('name'): a.text for a in shape.findall('attribute')})
                                if track_id is not None:
                                    attrs['track_id'] = track_id
                                occluded = shape.get('occluded') == '1' if shape.get('occluded') is not None else None
                                self.add(s, track_label, kind, g, attrs, occluded, dict(shape.attrib))
                            except (ValueError, TypeError, KeyError):
                                self.warnings.append(f"frame {shape.get('frame')}: skipped malformed {shape.tag} in track {track_id}")

                all_frame_indices = set(frames_dict.keys())
                if total_frames:
                    all_frame_indices.update(range(total_frames))
                all_frame_indices.update(media_by_frame.keys())

                for frame_idx in sorted(all_frame_indices):
                    if frame_idx not in frames_dict:
                        fname = media_by_frame[frame_idx].name if frame_idx in media_by_frame else f"frame_{frame_idx:06d}.PNG"
                        frames_dict[frame_idx] = sample(fname, orig_w, orig_h)
                    results.append(frames_dict[frame_idx])

        return results

class COCOParser(BaseAnnotationParser):
    def parse(self, paths, media):
        results = []
        for path in paths:
            data = json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=reject_json_constant)
            categories = {c['id']: c for c in data.get('categories',[])}
            images = {i['id']: sample(i['file_name'], i.get('width'), i.get('height')) for i in data['images']}
            for a in data.get('annotations',[]):
                if a.get('image_id') not in images:
                    self.warnings.append('Skipped annotation with unknown image_id')
                    continue
                s = images[a['image_id']]
                c = categories.get(a['category_id'], {'name': str(a['category_id'])})
                meta = {k: v for k,v in a.items() if k not in ('bbox','segmentation','keypoints')}
                try:
                    if a.get('keypoints'):
                        raw = a['keypoints']
                        if len(raw) % 3:
                            raise ValueError('Keypoints must be triples')
                        names = c.get('keypoints',[])
                        points = [dict(x=raw[i], y=raw[i+1], visibility=raw[i+2], **({'name':names[i//3]} if i//3 < len(names) else {})) for i in range(0,len(raw),3)]
                        self.add(s,c['name'],'KEYPOINT',{'points':points},metadata=meta)
                    elif a.get('segmentation'):
                        seg = a['segmentation']
                        if isinstance(seg, dict):
                            self.add(s,c['name'],'MASK',{'rle':seg},metadata=meta)
                            self.warnings.append(f"{s['file_name']}: RLE mask retained; mask metrics and rendering unavailable")
                        else:
                            for ring in seg:
                                if len(ring)%2:
                                    raise ValueError('Polygon must contain coordinate pairs')
                                self.add(s,c['name'],'POLYGON',{'points':[ring[i:i+2] for i in range(0,len(ring),2)]},metadata=meta)
                    elif 'bbox' in a:
                        x,y,w,h = a['bbox']
                        self.add(s,c['name'],'BBOX_2D',dict(x1=x,y1=y,x2=x+w,y2=y+h),metadata=meta)
                    else:
                        self.warnings.append(f"{s['file_name']}: annotation has no supported geometry")
                except (ValueError, TypeError, KeyError):
                    self.warnings.append(f"{s['file_name']}: skipped malformed COCO annotation")
            results.extend(images.values())
        return results

class YOLOParser(BaseAnnotationParser):
    def parse(self, paths, media):
        from PIL import Image
        results = []
        classes = []
        class_file = next((p for p in paths if p.name == 'classes.txt'), None)
        if class_file:
            classes = class_file.read_text(encoding='utf-8-sig').splitlines()
        for path in paths:
            if path.name == 'classes.txt':
                continue
            matches = [p for p in media if p.stem == path.stem and p.suffix.lower() in ('.jpg','.jpeg','.png')]
            if len(matches) != 1:
                raise ValueError(f'YOLO {path.name} requires exactly one matching image to denormalize coordinates')
            with Image.open(matches[0]) as image:
                w,h = image.size
            s = sample(matches[0].name,w,h)
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                try:
                    values = line.split()
                    if len(values) != 5:
                        raise ValueError('Only YOLO detection labels with 5 fields supported')
                    category = int(values[0])
                    x,y,bw,bh = map(float,values[1:])
                    if category < 0 or not all(0 <= v <= 1 for v in (x,y,bw,bh)):
                        raise ValueError('Invalid normalized coordinates')
                    self.add(s,classes[category] if category < len(classes) else str(category),'BBOX_2D',dict(x1=(x-bw/2)*w,y1=(y-bh/2)*h,x2=(x+bw/2)*w,y2=(y+bh/2)*h),metadata={'normalized_source':values})
                except (ValueError, IndexError):
                    self.warnings.append(f'{path.name}: skipped malformed YOLO line')
            results.append(s)
        return results

class KITTIParser(BaseAnnotationParser):
    def parse(self, paths, media):
        results = []
        for path in paths:
            if path.name == 'classes.txt':
                continue
            s = sample(path.stem)
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                try:
                    v = line.split()
                    if len(v) < 15:
                        raise ValueError('Expected KITTI 15 fields')
                    if v[0] == 'DontCare':
                        continue
                    trunc, occ = float(v[1]), int(v[2])
                    if not all(math.isfinite(float(x)) for x in v[1:15]) or not 0<=trunc<=1 or occ not in (0,1,2,3):
                        raise ValueError('Invalid KITTI metadata')
                    h,w,l,x,y,z,yaw = map(float,v[8:15])
                    self.add(s,v[0],'CUBOID_3D',{'position':dict(x=x,y=y,z=z),'dimensions':dict(height=h,width=w,length=l),'rotation':{'yaw':yaw}},occluded=occ > 0 if occ in (0,1,2) else None,metadata={'coordinate_system':'KITTI rectified camera; x right, y down, z forward; bottom center', 'truncation':trunc, 'occlusion_code':occ, 'bbox_2d':list(map(float,v[4:8])), 'alpha':float(v[3])})
                except (ValueError, IndexError):
                    self.warnings.append(f'{path.name}: skipped malformed KITTI line')
            results.append(s)
        return results

class BDD100KParser(BaseAnnotationParser):
    def parse(self, paths, media):
        results = []
        for path in paths:
            data = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=reject_json_constant)
            items = data if isinstance(data, list) else data.get('frames', data.get('images', []))
            for item in items:
                fname = item.get('name') or item.get('file_name')
                if not fname:
                    continue
                w = item.get('width', 1280)
                h = item.get('height', 720)
                s = sample(fname, w, h)
                for label_obj in item.get('labels', []):
                    cat = label_obj.get('category') or label_obj.get('label')
                    if not cat:
                        continue
                    box2d = label_obj.get('box2d')
                    if box2d:
                        try:
                            g = dict(
                                x1=float(box2d['x1']),
                                y1=float(box2d['y1']),
                                x2=float(box2d['x2']),
                                y2=float(box2d['y2'])
                            )
                            attrs = label_obj.get('attributes', {})
                            occ = attrs.get('occluded', False) if isinstance(attrs, dict) else False
                            self.add(s, cat, 'BBOX_2D', g, attributes=attrs if isinstance(attrs, dict) else {}, occluded=bool(occ), metadata=label_obj)
                        except (ValueError, KeyError):
                            self.warnings.append(f"{fname}: skipped malformed BDD100K box2d")
                results.append(s)
        return results

PARSERS = {'CVAT': CVATParser, 'COCO': COCOParser, 'BDD100K': BDD100KParser, 'YOLO': YOLOParser, 'KITTI': KITTIParser}

def detect_format(paths):
    suffixes = {p.suffix.lower() for p in paths}
    if '.xml' in suffixes:
        return 'CVAT'
    if '.json' in suffixes:
        for p in paths:
            if p.suffix.lower() == '.json':
                try:
                    head = p.read_text(encoding='utf-8-sig')[:4000]
                    if '"box2d"' in head or ('"labels"' in head and '"category"' in head):
                        return 'BDD100K'
                except Exception:
                    pass
        return 'COCO'
    for p in paths:
        if p.suffix.lower() == '.txt' and p.name != 'classes.txt':
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    return 'KITTI' if len(line.split()) >= 15 else 'YOLO'
    raise ValueError('Could not detect annotation format. Choose a format explicitly for empty TXT labels.')
