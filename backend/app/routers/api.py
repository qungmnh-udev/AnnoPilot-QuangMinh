from pathlib import Path
from typing import List, Optional, Union
import json
import shutil
import uuid
import zipfile
import xml.etree.ElementTree as ET
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database import get_db, UPLOADS
from app.models import Dataset, Sample, Annotation, Prediction, QCIssue, Settings
from app.schemas import ScoringSettings
from app.schemas.qc import QCConfig, IngestPredictionsRequest, QCIssueResolveRequest, CVATConfigRequest
from app.services.pipeline import store_files, import_dataset, sample_dict, dataset_dict
from app.services.qc_engine import (
    run_dataset_qc_audit,
    ingest_predictions_data,
    generate_benchmark_synthetic_predictions,
    run_pretrained_detector_service
)

router = APIRouter(prefix='/api')

def require(db, model, id):
    item = db.get(model, id)
    if item is None:
        raise HTTPException(404, model.__name__ + ' not found')
    return item

@router.get('/health')
def health():
    return {'status': 'ok', 'application': 'AnnoPilot-ModelQC'}

@router.get('/datasets')
def datasets(db: Session = Depends(get_db)):
    return [dataset_dict(d) for d in db.scalars(select(Dataset).order_by(Dataset.id.desc()))]

@router.post('/datasets/upload')
async def upload(
    name: str = Form(...),
    format: str = Form('AUTO'),
    task_type: str = Form('AUTO'),
    annotations: List[UploadFile] = File(...),
    media: Optional[List[Union[UploadFile, str]]] = File(None),
    db: Session = Depends(get_db)
):
    if not name.strip() or len(name) > 200:
        raise HTTPException(422, 'Dataset name must contain 1–200 characters')
    if task_type not in ('AUTO', 'BBOX_2D', 'SEGMENTATION', 'KEYPOINT', 'CUBOID_3D'):
        raise HTTPException(422, 'Unsupported annotation task')
    if any(isinstance(file, str) and file for file in (media or [])):
        raise HTTPException(422, 'Media must be uploaded files')
    folder = UPLOADS / uuid.uuid4().hex
    try:
        annotation_paths = await store_files(annotations, folder / 'annotations')
        media_paths = await store_files(
            [file for file in (media or []) if not isinstance(file, str) and file.filename],
            folder / 'media'
        )
        dataset = import_dataset(db, name, format, task_type, annotation_paths, media_paths)
        dataset.storage_key = folder.name
        db.commit()
        return dataset_dict(dataset)
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError, zipfile.BadZipFile) as e:
        db.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, 'Import failed: ' + str(e))
    except Exception:
        db.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise

@router.get('/datasets/{id}')
def get_dataset(id: int, db: Session = Depends(get_db)):
    return dataset_dict(require(db, Dataset, id))

@router.delete('/datasets/{id}')
def delete_dataset(id: int, db: Session = Depends(get_db)):
    dataset = require(db, Dataset, id)
    folders = {
        Path(s.media_path).relative_to(UPLOADS).parts[0]
        for s in dataset.samples
        if s.media_path and Path(s.media_path).is_relative_to(UPLOADS)
    }
    if dataset.storage_key:
        folders.add(dataset.storage_key)
    db.delete(dataset)
    db.commit()
    for folder in folders:
        shutil.rmtree(UPLOADS / folder, ignore_errors=True)
    return {'deleted': id}

@router.get('/datasets/{id}/samples')
def samples(id: int, db: Session = Depends(get_db)):
    ds = require(db, Dataset, id)
    return [sample_dict(s, dataset=ds) for s in ds.samples]

@router.get('/samples/{id}')
def sample_detail(id: int, db: Session = Depends(get_db)):
    s = require(db, Sample, id)
    ds = require(db, Dataset, s.dataset_id)
    return dict(
        **sample_dict(s, True, dataset=ds),
        dataset_id=s.dataset_id,
        source_format=ds.format
    )

@router.get('/samples/{id}/media')
def media_file(id: int, db: Session = Depends(get_db)):
    s = require(db, Sample, id)
    if not s.media_path or not Path(s.media_path).is_file():
        raise HTTPException(404, 'Media unavailable')
    return FileResponse(s.media_path)

@router.get('/settings')
def get_settings(db: Session = Depends(get_db)):
    row = db.get(Settings, 1)
    return row.config if row else {}

@router.put('/settings')
def save_settings(config: ScoringSettings, db: Session = Depends(get_db)):
    row = db.get(Settings, 1)
    if row:
        row.config = config.config or {}
    else:
        db.add(Settings(id=1, config=config.config or {}))
    db.commit()
    return config.config or {}

# --- Model-Assisted QC Endpoints (N2-04D) ---

@router.post('/datasets/{id}/qc/audit')
def trigger_qc_audit(id: int, config: Optional[QCConfig] = None, db: Session = Depends(get_db)):
    require(db, Dataset, id)
    try:
        report = run_dataset_qc_audit(db, id, config)
        return report
    except Exception as e:
        raise HTTPException(500, f"QC Audit failed: {str(e)}")

@router.get('/datasets/{id}/qc/report')
def get_qc_report(id: int, db: Session = Depends(get_db)):
    require(db, Dataset, id)
    report = run_dataset_qc_audit(db, id)
    return report

@router.post('/datasets/{id}/qc/predictions')
def ingest_predictions_json(
    id: int,
    payload: IngestPredictionsRequest,
    db: Session = Depends(get_db)
):
    require(db, Dataset, id)
    data = [p.model_dump() for p in payload.predictions]
    count = ingest_predictions_data(db, id, data, payload.source_model or 'pretrained')
    audit_report = run_dataset_qc_audit(db, id)
    return {
        'message': f'Successfully ingested {count} predictions',
        'predictions_count': count,
        'audit_summary': audit_report
    }

@router.post('/datasets/{id}/qc/predictions/upload')
async def upload_predictions_file(
    id: int,
    file: UploadFile = File(...),
    source_model: str = Form('yolov8-bdd100k'),
    db: Session = Depends(get_db)
):
    require(db, Dataset, id)
    content = await file.read()
    try:
        data = json.loads(content.decode('utf-8-sig'))
        if isinstance(data, dict):
            data = data.get('frames', data.get('images', [data]))
    except Exception as e:
        raise HTTPException(422, f"Failed to parse predictions JSON: {str(e)}")

    count = ingest_predictions_data(db, id, data, source_model)
    audit_report = run_dataset_qc_audit(db, id)
    return {
        'message': f'Successfully ingested {count} predictions',
        'predictions_count': count,
        'audit_summary': audit_report
    }

@router.post('/datasets/{id}/qc/synthetic-benchmark')
def run_synthetic_benchmark(
    id: int,
    missing_ratio: float = 0.15,
    wrong_class_ratio: float = 0.15,
    source_model: str = "synthetic_detector_v1",
    db: Session = Depends(get_db)
):
    require(db, Dataset, id)
    try:
        result = generate_benchmark_synthetic_predictions(
            db, id, missing_ratio, wrong_class_ratio, source_model
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Synthetic benchmark failed: {str(e)}")

@router.post('/datasets/{id}/qc/detect')
def run_model_detector(id: int, db: Session = Depends(get_db)):
    require(db, Dataset, id)
    try:
        result = run_pretrained_detector_service(db, id)
        return result
    except Exception as e:
        raise HTTPException(500, f"Model detection failed: {str(e)}")

@router.get('/datasets/{id}/qc/queue')
def get_qc_queue(id: int, include_clean: bool = False, db: Session = Depends(get_db)):
    dataset = require(db, Dataset, id)
    samples = []
    for s in dataset.samples:
        d = sample_dict(s, details=True, dataset=dataset)
        if include_clean or d['qc_issue_count'] > 0:
            samples.append(d)
    # Sort prioritized queue by qc_score descending
    samples.sort(key=lambda x: (-x['qc_score'], x['id']))
    return samples

@router.put('/datasets/{id}/cvat-config')
def update_cvat_config(id: int, body: CVATConfigRequest, db: Session = Depends(get_db)):
    dataset = require(db, Dataset, id)
    if body.cvat_base_url is not None:
        dataset.cvat_base_url = body.cvat_base_url
    if body.cvat_task_id is not None:
        dataset.cvat_task_id = body.cvat_task_id
    if body.cvat_job_id is not None:
        dataset.cvat_job_id = body.cvat_job_id
    db.commit()
    return dataset_dict(dataset)

@router.get('/qc/issues/{issue_id}/cvat-link')
def get_cvat_link(issue_id: int, db: Session = Depends(get_db)):
    issue = db.get(QCIssue, issue_id)
    if not issue:
        raise HTTPException(404, "QC Issue not found")
    sample = require(db, Sample, issue.sample_id)
    dataset = require(db, Dataset, sample.dataset_id)
    frame_number = getattr(sample, 'frame_number', None)
    if frame_number is None:
        from app.services.pipeline import extract_frame_number
        frame_number = extract_frame_number(sample.file_name, sample.id - 1)

    base = (dataset.cvat_base_url or 'http://localhost:8080').rstrip('/')
    if dataset.cvat_task_id:
        if dataset.cvat_job_id:
            url = f"{base}/tasks/{dataset.cvat_task_id}/jobs/{dataset.cvat_job_id}?frame={frame_number}"
        else:
            url = f"{base}/tasks/{dataset.cvat_task_id}?frame={frame_number}"
    else:
        url = f"{base}/tasks?frame={frame_number}"

    return {
        'issue_id': issue.id,
        'sample_id': sample.id,
        'frame_number': frame_number,
        'cvat_url': url
    }

@router.put('/qc/issues/{issue_id}')
def resolve_qc_issue(issue_id: int, body: QCIssueResolveRequest, db: Session = Depends(get_db)):
    issue = db.get(QCIssue, issue_id)
    if not issue:
        if body.sample_id:
            query = select(QCIssue).where(QCIssue.sample_id == body.sample_id)
            if body.issue_type:
                query = query.where(QCIssue.issue_type == body.issue_type)
            if body.suggested_label:
                query = query.where(QCIssue.suggested_label == body.suggested_label)
            fallback = db.scalars(query).first()
            if fallback:
                issue = fallback
        if not issue:
            raise HTTPException(404, f"QC Issue #{issue_id} not found. Vấn đề có thể đã được cập nhật sau lần chạy đối soát mới; vui lòng làm mới mẫu.")

    sample = require(db, Sample, issue.sample_id)
    normalized_status = body.status

    # If resolved via 1-Click action or verified in CVAT, update the ground truth annotation in DB
    if body.status in ('ACCEPTED', 'RESOLVED'):
        if issue.issue_type == 'MISSING_OBJECT':
            new_annotation = Annotation(
                sample_id=sample.id,
                label=issue.suggested_label,
                shape_type='BBOX_2D',
                geometry=issue.location,
                attributes={'qc_verified': True, 'resolved_from_issue_id': issue.id},
                occluded=False,
                source_metadata={'origin': 'model_assisted_qc'}
            )
            db.add(new_annotation)
        elif issue.issue_type == 'WRONG_CLASS':
            if issue.annotation_id:
                ann = db.get(Annotation, issue.annotation_id)
                if ann:
                    if ann.attributes is None:
                        ann.attributes = {}
                    ann.attributes['previous_label'] = ann.label
                    ann.attributes['qc_verified'] = True
                    ann.label = issue.suggested_label
    elif body.status == 'PENDING':
        # Undo: remove added annotations or restore previous label
        for ann in list(sample.annotations):
            attrs = ann.attributes or {}
            if attrs.get('resolved_from_issue_id') == issue.id:
                db.delete(ann)
            elif attrs.get('previous_label') and issue.annotation_id == ann.id:
                ann.label = attrs['previous_label']
                attrs.pop('previous_label', None)
                attrs.pop('qc_verified', None)

    issue.status = normalized_status
    issue.reviewer_note = body.reviewer_note or ''
    db.commit()

    return {
        'id': issue.id,
        'status': issue.status,
        'reviewer_note': issue.reviewer_note,
        'issue_type': issue.issue_type,
        'suggested_label': issue.suggested_label
    }
