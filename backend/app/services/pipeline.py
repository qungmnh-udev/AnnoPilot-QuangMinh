from collections import Counter
from pathlib import Path, PurePosixPath
import math
import random
import shutil
import zipfile
from PIL import Image
from sqlalchemy import select
from app.database import UPLOADS
from app.models import Dataset, Sample, Annotation, AnalysisResult, Settings, SamplingRun, SampleSelection
from app.parsers import PARSERS, detect_format, task_type
from app.analyzers import analyze_sample
from app.schemas import ScoringSettings

MAX_UPLOAD = 512 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024

def get_settings(db):
    row = db.get(Settings,1)
    return ScoringSettings(**row.config) if row else ScoringSettings()

def analyze_dataset(db,dataset,settings=None):
    settings = settings or get_settings(db)
    counts = Counter(a.label for sample in dataset.samples for a in sample.annotations)
    total = sum(counts.values())
    rare = {k for k,v in counts.items() if total and v/total < settings.rare_threshold}
    for sample in dataset.samples:
        values = analyze_sample(sample,rare,settings)
        if sample.analysis:
            for k,v in values.items():
                setattr(sample.analysis,k,v)
        else:
            sample.analysis = AnalysisResult(**values)

async def store_files(files,destination):
    destination.mkdir(parents=True,exist_ok=True)
    total = 0
    expanded_total = 0
    extracted_count = 0
    if len(files)>10000:
        raise ValueError('At most 10,000 files per upload group')
    for index,upload in enumerate(files):
        name = Path((upload.filename or '').replace('\\','/')).name
        if not name:
            raise ValueError('Upload file has no name')
        # Separate uploads so duplicate basenames cannot overwrite one another.
        folder = destination / str(index)
        folder.mkdir()
        target = folder / name
        with target.open('wb') as output:
            while True:
                chunk = await upload.read(1024*1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise ValueError('Uploads exceed 512 MiB per file group')
                output.write(chunk)
        if target.suffix.lower() == '.zip':
            extracted = folder / 'extracted'
            with zipfile.ZipFile(target) as archive:
                infos = archive.infolist()
                expanded_total += sum(i.file_size for i in infos)
                extracted_count += len(infos)
                if extracted_count>10000 or expanded_total>MAX_EXPANDED:
                    raise ValueError('Archive exceeds extraction limit (10,000 files / 2 GiB)')
                for info in infos:
                    relative = PurePosixPath(info.filename.replace('\\','/'))
                    if relative.is_absolute() or '..' in relative.parts or any(':' in part for part in relative.parts):
                        raise ValueError('Unsafe ZIP path')
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('ZIP symbolic links are not supported')
                    archive.extract(info,extracted)
            target.unlink()
    return sorted(p for p in destination.rglob('*') if p.is_file())

def match_media(name,media):
    name = name.replace('\\','/')
    exact = [p for p in media if p.as_posix().endswith('/'+name)]
    if len(exact)==1:
        return exact[0],None
    matches = [p for p in media if p.name == Path(name).name]
    if not matches:
        matches = [p for p in media if p.stem == Path(name).stem]
        images = [p for p in matches if p.suffix.lower() in ('.jpg','.jpeg','.png')]
        if images:
            matches = images
    if len(matches)==1:
        return matches[0],None
    return None,'Ambiguous media match' if matches else 'Media unavailable'

def import_dataset(db,name,requested_format,requested_task,annotation_paths,media):
    fmt = detect_format(annotation_paths) if requested_format=='AUTO' else requested_format
    if fmt not in PARSERS:
        raise ValueError('Unsupported annotation format')
    suffix = {'CVAT':'.xml','COCO':'.json','BDD100K':'.json','YOLO':'.txt','KITTI':'.txt'}[fmt]
    paths = [p for p in annotation_paths if p.suffix.lower()==suffix]
    if not paths:
        raise ValueError('No annotation files found for '+fmt)
    parser = PARSERS[fmt]()
    normalized = parser.parse(paths,media)
    if not normalized:
        raise ValueError('No samples found in annotations')
    if len(normalized)>50000:
        raise ValueError('Local MVP supports at most 50,000 samples per import')
    names = [s['file_name'] for s in normalized]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate annotation sample names; upload a single coherent dataset')
    all_annotations = [a for s in normalized for a in s['annotations']]
    if not all_annotations:
        parser.warnings.append('Dataset contains zero valid annotations; scores describe available media only or empty annotations.')
    detected = task_type(all_annotations)
    if requested_task != 'AUTO' and detected not in (requested_task,'UNKNOWN'):
        raise ValueError(f'Requested task {requested_task} differs from detected {detected}')
    dataset = Dataset(name=name.strip(),format=fmt,task_type=detected,warnings=[])
    db.add(dataset)
    db.flush()
    for source in normalized:
        annotations = source.pop('annotations')
        path,warning = match_media(source['file_name'],media)
        if warning:
            parser.warnings.append(source['file_name']+': '+warning)
        if path and path.suffix.lower() in ('.jpg','.jpeg','.png'):
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width,height = image.size
                if source['width'] and source['height'] and (width,height)!=(source['width'],source['height']):
                    parser.warnings.append(source['file_name']+': media dimensions differ from annotation dimensions; overlays use annotation coordinates')
                source['width'] = source['width'] or width
                source['height'] = source['height'] or height
            except (OSError,ValueError,Image.DecompressionBombError):
                parser.warnings.append(source['file_name']+': image could not be decoded')
                path = None
        sample = Sample(**source, dataset_id=dataset.id,media_path=str(path) if path else None, task_type=task_type(annotations))
        sample.annotations = [Annotation(**a) for a in annotations]
        dataset.samples.append(sample)
    dataset.warnings = parser.warnings
    analyze_dataset(db,dataset)
    return dataset

def sample_dict(s,details=False):
    a = s.analysis
    qc_issues = getattr(s, 'qc_issues', [])
    qc_score, qc_severity = 0.0, 'CLEAN'
    if qc_issues:
        scores = [i.qc_score for i in qc_issues]
        max_score = max(scores) if scores else 0.0
        boost = 0.04 * min(len(scores)-1, 5) if len(scores)>1 else 0.0
        qc_score = min(1.0, round(max_score + boost, 4))
        qc_severity = 'HIGH' if qc_score >= 0.70 else ('MEDIUM' if qc_score >= 0.45 else 'LOW')
    preds = getattr(s, 'predictions', [])
    result = dict(id=s.id,file_name=s.file_name,task_type=s.task_type,width=s.width,height=s.height,annotation_count=len(s.annotations),prediction_count=len(preds),qc_score=qc_score,qc_severity=qc_severity,qc_issue_count=len(qc_issues),media_url=f'/api/samples/{s.id}/media' if s.media_path else None,media_kind=Path(s.media_path).suffix.lower() if s.media_path else None,annotation_difficulty=a.annotation_difficulty if a else None,visual_difficulty=a.visual_difficulty if a else None,overall_difficulty=a.overall_difficulty if a else None,level=a.level if a else 'UNAVAILABLE',rare_classes=a.details.get('rare_classes',[]) if a else [])
    if details:
        result['analysis'] = a.details if a else None
        result['annotations'] = [dict(id=x.id,label=x.label,shape_type=x.shape_type,geometry=x.geometry,attributes=x.attributes,occluded=x.occluded,metadata=x.source_metadata) for x in s.annotations]
        result['predictions'] = [dict(id=p.id,label=p.label,geometry=p.geometry,confidence=p.confidence,source_model=p.source_model) for p in preds]
        result['qc_issues'] = [dict(id=i.id,issue_type=i.issue_type,location=i.location,human_label=i.human_label,suggested_label=i.suggested_label,annotation_id=i.annotation_id,prediction_id=i.prediction_id,qc_score=i.qc_score,evidence=i.evidence,status=i.status,reviewer_note=i.reviewer_note) for i in qc_issues]
    return result

def dataset_dict(dataset):
    samples = dataset.samples
    counts = Counter(a.label for s in samples for a in s.annotations)
    levels = Counter(s.analysis.level for s in samples if s.analysis)
    scores = [s.analysis.overall_difficulty for s in samples if s.analysis and s.analysis.overall_difficulty is not None]
    rare = sorted({c for s in samples if s.analysis for c in s.analysis.details.get('rare_classes',[])})
    return dict(id=dataset.id,name=dataset.name,format=dataset.format,task_type=dataset.task_type,revision=dataset.revision,created_at=dataset.created_at.isoformat(),sample_count=len(samples),annotation_count=sum(counts.values()),average_difficulty=round(sum(scores)/len(scores),2) if scores else None,levels=dict(levels),classes=dict(counts),annotation_types=dict(Counter(a.shape_type for s in samples for a in s.annotations)),rare_classes=rare,missing_media=sum(not s.media_path for s in samples),warnings=dataset.warnings)

def edge_score(s):
    d = s.analysis.details
    values = [f['value'] for v in d['per_type'].values() for k,f in v['features'].items() if k!='count' and f['available']]
    values += [f['value'] for k,f in d['visual']['features'].items() if k in ('blur','exposure','low_contrast') and f['available']]
    return sum(values)/len(values) if values else 0

def choose_samples(samples,request):
    n = min(len(samples), math.ceil(len(samples)*request.budget/100) if request.mode=='percentage' else int(request.budget))
    keys = ['random','difficulty','edge']
    exact = {k:n*request.weights[k] for k in keys}
    quotas = {k:int(exact[k]) for k in keys}
    for k in sorted(keys,key=lambda k:-(exact[k]-quotas[k]))[:n-sum(quotas.values())]:
        quotas[k]+=1
    pool = sorted(samples,key=lambda s:s.id)
    random.Random(request.seed).shuffle(pool)
    orders = {'random':pool,'difficulty':sorted([s for s in samples if s.analysis.overall_difficulty is not None],key=lambda s:(-s.analysis.overall_difficulty,s.id)),'edge':sorted([s for s in samples if edge_score(s)>0],key=lambda s:(-edge_score(s),s.id))}
    chosen = {}
    for k in keys:
        for s in orders[k]:
            if sum(reason==k for reason in chosen.values())>=quotas[k]:
                break
            if s.id not in chosen:
                chosen[s.id]=k
    # If real edge signals or available scores cannot fill a bucket, fill from
    # seeded coverage and record that redistribution explicitly.
    for s in pool:
        if len(chosen)>=n:
            break
        if s.id not in chosen:
            chosen[s.id]='coverage_fallback'
    return chosen

def balance(samples,reviewers):
    if any(s.analysis.overall_difficulty is None for s in samples):
        raise ValueError('Selected samples have unavailable difficulty; adjust weights or supply media and recalculate before balancing')
    loads = {r.id:0.0 for r in reviewers}
    counts = {r.id:0 for r in reviewers}
    order = {r.id:i for i,r in enumerate(reviewers)}
    result = {}
    for s in sorted(samples,key=lambda s:(-(s.analysis.overall_difficulty or 0),s.id)):
        who = min(loads,key=lambda r:(loads[r],counts[r],order[r]))
        result[s.id]=who
        loads[who]+=s.analysis.overall_difficulty or 0
        counts[who]+=1
    return result
