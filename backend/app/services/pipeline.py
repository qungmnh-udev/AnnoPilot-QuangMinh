from collections import Counter
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from PIL import Image
from sqlalchemy import select
from app.database import UPLOADS
from app.models import Dataset, Sample, Annotation, Prediction, QCIssue
from app.parsers import PARSERS, detect_format, task_type

MAX_UPLOAD = 512 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024

async def store_files(files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    expanded_total = 0
    extracted_count = 0
    if len(files) > 10000:
        raise ValueError('At most 10,000 files per upload group')
    for index, upload in enumerate(files):
        name = Path((upload.filename or '').replace('\\', '/')).name
        if not name:
            raise ValueError('Upload file has no name')
        folder = destination / str(index)
        folder.mkdir()
        target = folder / name
        with target.open('wb') as output:
            while True:
                chunk = await upload.read(1024 * 1024)
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
                if extracted_count > 10000 or expanded_total > MAX_EXPANDED:
                    raise ValueError('Archive exceeds extraction limit (10,000 files / 2 GiB)')
                for info in infos:
                    relative = PurePosixPath(info.filename.replace('\\', '/'))
                    if relative.is_absolute() or '..' in relative.parts or any(':' in part for part in relative.parts):
                        raise ValueError('Unsafe ZIP path')
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('ZIP symbolic links are not supported')
                    archive.extract(info, extracted)
            target.unlink()
    return sorted(p for p in destination.rglob('*') if p.is_file())

def match_media(name, media):
    name = name.replace('\\', '/')
    exact = [p for p in media if p.as_posix().endswith('/' + name)]
    if len(exact) == 1:
        return exact[0], None
    matches = [p for p in media if p.name == Path(name).name]
    if not matches:
        matches = [p for p in media if p.stem == Path(name).stem]
        images = [p for p in matches if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if images:
            matches = images
    if len(matches) == 1:
        return matches[0], None
    return None, 'Ambiguous media match' if matches else 'Media unavailable'

def import_dataset(db, name, requested_format, requested_task, annotation_paths, media):
    image_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not media:
        media = [p for p in annotation_paths if p.suffix.lower() in image_suffixes]
    actual_annotation_paths = [p for p in annotation_paths if p.suffix.lower() not in image_suffixes]
    if not actual_annotation_paths:
        actual_annotation_paths = annotation_paths

    fmt = detect_format(actual_annotation_paths) if requested_format == 'AUTO' else requested_format
    if fmt not in PARSERS:
        raise ValueError('Unsupported annotation format')
    suffix = {'CVAT': '.xml', 'COCO': '.json', 'BDD100K': '.json', 'YOLO': '.txt', 'KITTI': '.txt'}[fmt]
    paths = [p for p in actual_annotation_paths if p.suffix.lower() == suffix]
    if not paths:
        raise ValueError('No annotation files found for ' + fmt)
    parser = PARSERS[fmt]()
    normalized = parser.parse(paths, media)
    if not normalized:
        raise ValueError('No samples found in annotations')
    if len(normalized) > 50000:
        raise ValueError('Local MVP supports at most 50,000 samples per import')
    names = [s['file_name'] for s in normalized]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate annotation sample names; upload a single coherent dataset')
    all_annotations = [a for s in normalized for a in s['annotations']]
    if not all_annotations:
        parser.warnings.append('Dataset contains zero valid annotations.')
    detected = task_type(all_annotations)
    if requested_task != 'AUTO' and detected not in (requested_task, 'UNKNOWN'):
        raise ValueError(f'Requested task {requested_task} differs from detected {detected}')
    dataset = Dataset(name=name.strip(), format=fmt, task_type=detected, warnings=[])
    if getattr(parser, 'cvat_task_id', None) is not None:
        dataset.cvat_task_id = parser.cvat_task_id
    if getattr(parser, 'cvat_job_id', None) is not None:
        dataset.cvat_job_id = parser.cvat_job_id
    db.add(dataset)
    db.flush()
    for source in normalized:
        annotations = source.pop('annotations')
        path, warning = match_media(source['file_name'], media)
        if warning:
            parser.warnings.append(source['file_name'] + ': ' + warning)
        if path and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                if source['width'] and source['height'] and (width, height) != (source['width'], source['height']):
                    parser.warnings.append(source['file_name'] + ': media dimensions differ from annotation dimensions')
                source['width'] = source['width'] or width
                source['height'] = source['height'] or height
            except (OSError, ValueError, Image.DecompressionBombError):
                parser.warnings.append(source['file_name'] + ': image could not be decoded')
                path = None
        sample = Sample(**source, dataset_id=dataset.id, media_path=str(path) if path else None, task_type=task_type(annotations))
        sample.frame_number = extract_frame_number(source['file_name'])
        sample.annotations = [Annotation(**a) for a in annotations]
        dataset.samples.append(sample)
    dataset.warnings = parser.warnings
    return dataset

import re

def extract_frame_number(file_name: str, fallback_id: int = 0) -> int:
    digits = re.findall(r'\d+', Path(file_name).stem)
    if digits:
        try:
            return int(digits[-1])
        except ValueError:
            pass
    return max(0, fallback_id)

def sample_dict(s, details=False, dataset=None):
    qc_issues = getattr(s, 'qc_issues', [])
    qc_score, qc_severity = 0.0, 'CLEAN'
    if qc_issues:
        scores = [i.qc_score for i in qc_issues]
        max_score = max(scores) if scores else 0.0
        boost = 0.04 * min(len(scores) - 1, 5) if len(scores) > 1 else 0.0
        qc_score = min(1.0, round(max_score + boost, 4))
        qc_severity = 'HIGH' if qc_score >= 0.70 else ('MEDIUM' if qc_score >= 0.45 else 'LOW')
    preds = getattr(s, 'predictions', [])
    
    frame_number = getattr(s, 'frame_number', None)
    if frame_number is None:
        frame_number = extract_frame_number(s.file_name, (s.id or 1) - 1)
        
    ds = dataset or getattr(s, 'dataset', None)
    base = (getattr(ds, 'cvat_base_url', None) or 'http://localhost:8080').rstrip('/')
    if ds and getattr(ds, 'cvat_task_id', None):
        if getattr(ds, 'cvat_job_id', None):
            cvat_url = f"{base}/tasks/{ds.cvat_task_id}/jobs/{ds.cvat_job_id}?frame={frame_number}"
        else:
            cvat_url = f"{base}/tasks/{ds.cvat_task_id}?frame={frame_number}"
    else:
        cvat_url = f"{base}/tasks?frame={frame_number}"

    result = dict(
        id=s.id,
        file_name=s.file_name,
        frame_number=frame_number,
        cvat_url=cvat_url,
        task_type=s.task_type,
        width=s.width,
        height=s.height,
        annotation_count=len(s.annotations),
        prediction_count=len(preds),
        qc_score=qc_score,
        qc_severity=qc_severity,
        qc_issue_count=len(qc_issues),
        media_url=f'/api/samples/{s.id}/media' if s.media_path else None,
        media_kind=Path(s.media_path).suffix.lower() if s.media_path else None,
        annotation_difficulty=None,
        visual_difficulty=None,
        overall_difficulty=None,
        level='CLEAN' if qc_severity == 'CLEAN' else qc_severity,
        rare_classes=[]
    )
    if details:
        result['analysis'] = None
        result['annotations'] = [
            dict(
                id=x.id,
                label=x.label,
                shape_type=x.shape_type,
                geometry=x.geometry,
                attributes=x.attributes,
                occluded=x.occluded,
                metadata=x.source_metadata
            ) for x in s.annotations
        ]
        result['predictions'] = [
            dict(
                id=p.id,
                label=p.label,
                geometry=p.geometry,
                confidence=p.confidence,
                source_model=p.source_model
            ) for p in preds
        ]
        result['qc_issues'] = [
            dict(
                id=i.id,
                issue_type=i.issue_type,
                location=i.location,
                human_label=i.human_label,
                suggested_label=i.suggested_label,
                annotation_id=i.annotation_id,
                prediction_id=i.prediction_id,
                qc_score=i.qc_score,
                evidence=i.evidence,
                frame_number=frame_number,
                cvat_url=cvat_url,
                status=i.status,
                reviewer_note=i.reviewer_note
            ) for i in qc_issues
        ]
    return result

def dataset_dict(dataset):
    samples = dataset.samples
    counts = Counter(a.label for s in samples for a in s.annotations)
    return dict(
        id=dataset.id,
        name=dataset.name,
        format=dataset.format,
        task_type=dataset.task_type,
        revision=dataset.revision,
        cvat_task_id=getattr(dataset, 'cvat_task_id', None),
        cvat_job_id=getattr(dataset, 'cvat_job_id', None),
        cvat_base_url=getattr(dataset, 'cvat_base_url', 'http://localhost:8080'),
        created_at=dataset.created_at.isoformat(),
        sample_count=len(samples),
        annotation_count=sum(counts.values()),
        average_difficulty=None,
        levels={'EASY': len(samples), 'MEDIUM': 0, 'HARD': 0},
        classes=dict(counts),
        annotation_types=dict(Counter(a.shape_type for s in samples for a in s.annotations)),
        rare_classes=[],
        missing_media=sum(not s.media_path for s in samples),
        warnings=dataset.warnings
    )
