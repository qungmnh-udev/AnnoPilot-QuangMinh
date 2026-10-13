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
from sqlalchemy.orm import Session, selectinload
from app.database import get_db, UPLOADS
from app.models import Dataset, Sample, Annotation, Prediction, QCIssue, Settings
from app.schemas import ScoringSettings
from app.schemas.qc import (
    QCConfig,
    IngestPredictionsRequest,
    QCIssueResolveRequest,
    QCBatchResolveRequest,
    CVATConfigRequest,
    StartQueueRequest,
    FrameQueueStatusResponse,
    FrameAuditResponse,
    CVATSyncResponse,
    CVATVerifyFrameResponse,
    CVATPushRequest,
    CVATPushResponse,
    EnsembleRunRequest,
    DatasetModelsResponse,
    EnsembleResponse,
    TriadRunRequest,
    TriadRunResponse
)
from app.services.pipeline import store_files, import_dataset, sample_dict, dataset_dict
from app.services.qc_engine import (
    run_dataset_qc_audit,
    get_dataset_qc_report,
    invalidate_qc_report_cache,
    audit_single_frame,
    queue_manager,
    sync_cvat_dataset_annotations,
    verify_and_sync_sample_from_cvat,
    push_dataset_annotations_to_cvat,
    ingest_predictions_data,
    generate_benchmark_synthetic_predictions,
    run_pretrained_detector_service
)
from app.services.ensemble import (
    get_dataset_prediction_models,
    run_ensemble_from_existing_predictions,
    load_predictions_from_json,
    run_ensemble_and_store,
)
from app.services.triad_detector import (
    TRIAD_SPECS,
    run_dataset_triad_inference,
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
    report = get_dataset_qc_report(db, id)
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

# --- Multi-Model Ensemble Pre-labeling Endpoints ---

@router.get('/datasets/{id}/ensemble/models', response_model=DatasetModelsResponse)
def get_ensemble_models(id: int, db: Session = Depends(get_db)):
    """
    Get all model prediction sources currently ingested in this dataset,
    along with prediction counts, to configure ensemble fusion.
    """
    require(db, Dataset, id)
    return get_dataset_prediction_models(db, id)

@router.post('/datasets/{id}/ensemble/run', response_model=EnsembleResponse)
def run_ensemble_endpoint(
    id: int,
    payload: Optional[EnsembleRunRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Execute Weighted Box Fusion (WBF) ensemble combining predictions from
    multiple models already stored in the dataset, categorize into Tri-Tier
    consensus, and optionally re-audit dataset QC.
    """
    require(db, Dataset, id)
    p = payload or EnsembleRunRequest()
    try:
        result = run_ensemble_from_existing_predictions(
            db=db,
            dataset_id=id,
            model_names=p.model_names,
            model_weights=p.model_weights,
            iou_threshold=p.iou_threshold,
            confidence_threshold=p.confidence_threshold,
            high_consensus_conf=p.high_consensus_conf,
            high_consensus_iou=p.high_consensus_iou,
            ensemble_source_model=p.ensemble_source_model,
            run_audit=p.run_audit,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Ensemble execution failed: {str(e)}")

@router.post('/datasets/{id}/ensemble/upload-and-run')
async def upload_multi_model_and_run_ensemble(
    id: int,
    file: UploadFile = File(...),
    model_name: str = Form("external_model"),
    iou_threshold: float = Form(0.55),
    confidence_threshold: float = Form(0.30),
    ensemble_source_model: str = Form("ensemble_wbf_v1"),
    db: Session = Depends(get_db)
):
    """
    Upload predictions for an additional model (e.g. YOLO, RT-DETR, Faster R-CNN)
    and immediately run WBF ensemble across all available models.
    """
    require(db, Dataset, id)
    content = await file.read()
    try:
        data = json.loads(content.decode('utf-8-sig'))
        if isinstance(data, dict):
            data = data.get('frames', data.get('images', [data]))
    except Exception as e:
        raise HTTPException(422, f"Failed to parse predictions JSON: {str(e)}")

    # Ingest the new model predictions
    ingest_predictions_data(db, id, data, model_name)

    # Run ensemble
    try:
        result = run_ensemble_from_existing_predictions(
            db=db,
            dataset_id=id,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            ensemble_source_model=ensemble_source_model,
            run_audit=True,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Ensemble execution failed: {str(e)}")

# --- Option B Triad Endpoints ---

@router.get('/ensemble/triad/specs')
def get_triad_specs():
    """
    Get architectural specs for Option B Triad models:
    YOLOv8n (1-Stage CNN), Faster R-CNN MobileNetV3 (2-Stage CNN), RT-DETR-l (Vision Transformer).
    """
    return TRIAD_SPECS

@router.post('/datasets/{id}/ensemble/run-triad', response_model=TriadRunResponse)
def run_triad_ensemble_endpoint(
    id: int,
    payload: Optional[TriadRunRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Execute Option B Triad Inference & WBF Ensemble:
      1. Runs YOLOv8n + Faster R-CNN MobileNetV3 + RT-DETR on dataset frames
      2. Persists normalized BDD100K predictions
      3. Automatically executes Weighted Box Fusion with Tri-Tier Consensus
    """
    require(db, Dataset, id)
    p = payload or TriadRunRequest()
    try:
        result = run_dataset_triad_inference(
            db=db,
            dataset_id=id,
            models=p.models,
            conf_thresh=p.conf_thresh,
            sample_limit=p.sample_limit,
            auto_ensemble=p.auto_ensemble,
            weights=p.weights,
            iou_thresh=p.iou_thresh
        )
        result["triad_specs"] = TRIAD_SPECS
        return result
    except Exception as e:
        raise HTTPException(500, f"Triad inference execution failed: {str(e)}")

@router.get('/datasets/{id}/qc/queue')
def get_qc_queue(id: int, include_clean: bool = False, db: Session = Depends(get_db)):
    dataset = require(db, Dataset, id)
    samples_db = db.scalars(
        select(Sample)
        .where(Sample.dataset_id == id)
        .options(
            selectinload(Sample.annotations),
            selectinload(Sample.predictions),
            selectinload(Sample.qc_issues)
        )
    ).all()
    samples = []
    for s in samples_db:
        d = sample_dict(s, details=True, dataset=dataset)
        if include_clean or d['qc_issue_count'] > 0:
            samples.append(d)
    # Sort prioritized queue by qc_score descending
    samples.sort(key=lambda x: (-x['qc_score'], x['id']))
    return samples

# --- Streaming Incremental Frame Processing & Two-Way CVAT Sync (P0) ---

@router.post('/datasets/{id}/qc/frame/{sample_id}/audit', response_model=FrameAuditResponse)
def audit_single_frame_endpoint(
    id: int,
    sample_id: int,
    conf_thresh: float = 0.35,
    force_repredict: bool = False,
    db: Session = Depends(get_db)
):
    """
    On-demand single-frame audit. Executes detection (if required/empty), matches against annotations,
    updates DB issues, and immediately returns the prioritized summary for this frame.
    """
    require(db, Dataset, id)
    sample = require(db, Sample, sample_id)
    if sample.dataset_id != id:
        raise HTTPException(400, "Sample does not belong to the specified dataset")
    try:
        summary = audit_single_frame(
            db,
            sample_id,
            run_detector=force_repredict or not sample.predictions,
            conf_thresh=conf_thresh
        )
        return {
            'sample_summary': summary,
            'predictions_count': len(sample.predictions)
        }
    except Exception as e:
        raise HTTPException(500, f"Single frame audit failed: {str(e)}")

@router.post('/datasets/{id}/qc/queue/start', response_model=FrameQueueStatusResponse)
def start_qc_queue(
    id: int,
    payload: Optional[StartQueueRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Start background frame-by-frame streaming QC worker.
    Processes one frame at a time, allowing live progress updates.
    """
    require(db, Dataset, id)
    conf_thresh = payload.conf_thresh if (payload and payload.conf_thresh) else 0.35
    sample_ids = payload.sample_ids if payload else None
    force_repredict = payload.force_repredict if payload and payload.force_repredict is not None else False

    status = queue_manager.start_queue(
        dataset_id=id,
        sample_ids=sample_ids,
        conf_thresh=conf_thresh,
        force_repredict=force_repredict
    )
    return status

@router.get('/datasets/{id}/qc/queue/status', response_model=FrameQueueStatusResponse)
def get_qc_queue_status(id: int, db: Session = Depends(get_db)):
    """
    Poll live progress of the incremental QC queue worker.
    """
    require(db, Dataset, id)
    return queue_manager.get_status(id)

@router.post('/datasets/{id}/qc/queue/stop', response_model=FrameQueueStatusResponse)
def stop_qc_queue(id: int, db: Session = Depends(get_db)):
    """
    Pause or stop the running background QC queue.
    """
    require(db, Dataset, id)
    queue_manager.stop_queue(id)
    return queue_manager.get_status(id)

@router.post('/datasets/{id}/qc/cvat-sync', response_model=CVATSyncResponse)
async def sync_cvat_annotations_endpoint(
    id: int,
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Synchronize annotations from CVAT (the single source of truth).
    Accepts updated CVAT XML or BDD100K JSON export, updates annotations,
    re-audits affected frames, and auto-resolves fixed issues.
    """
    require(db, Dataset, id)
    content = None
    if file and file.filename:
        content = await file.read()
    try:
        result = sync_cvat_dataset_annotations(db, id, content=content)
        return result
    except Exception as e:
        raise HTTPException(500, f"CVAT Sync failed: {str(e)}")

@router.post('/datasets/{id}/qc/frame/{sample_id}/cvat-verify', response_model=CVATVerifyFrameResponse)
def verify_cvat_frame_endpoint(
    id: int,
    sample_id: int,
    db: Session = Depends(get_db)
):
    """
    Re-audit an individual frame on-demand after reviewer made corrections in CVAT.
    Validates whether flagged issues have been properly fixed.
    """
    require(db, Dataset, id)
    sample = require(db, Sample, sample_id)
    if sample.dataset_id != id:
        raise HTTPException(400, "Sample does not belong to the specified dataset")
    try:
        result = verify_and_sync_sample_from_cvat(db, sample_id)
        return result
    except Exception as e:
        raise HTTPException(500, f"CVAT frame verification failed: {str(e)}")

@router.post('/datasets/{id}/qc/cvat-push', response_model=CVATPushResponse)
def push_to_cvat_endpoint(
    id: int,
    payload: Optional[CVATPushRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Push verified/human-in-the-loop annotations from AnnoPilot directly into CVAT Task/Job
    using the CVAT REST API with Personal Access Token (PAT).
    """
    require(db, Dataset, id)
    token = payload.token if payload else None
    cvat_base_url = payload.cvat_base_url if payload else None
    cvat_task_id = payload.cvat_task_id if payload else None
    cvat_job_id = payload.cvat_job_id if payload else None
    mode = payload.mode if payload and payload.mode else "create"
    target_source = payload.target_source if payload and payload.target_source else "merged"

    try:
        result = push_dataset_annotations_to_cvat(
            db,
            id,
            token=token,
            cvat_base_url=cvat_base_url,
            cvat_task_id=cvat_task_id,
            cvat_job_id=cvat_job_id,
            mode=mode,
            target_source=target_source
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"CVAT Push failed: {str(e)}")

@router.put('/datasets/{id}/cvat-config')
def update_cvat_config(id: int, body: CVATConfigRequest, db: Session = Depends(get_db)):
    dataset = require(db, Dataset, id)
    if body.cvat_base_url is not None:
        dataset.cvat_base_url = body.cvat_base_url
    if body.cvat_task_id is not None:
        dataset.cvat_task_id = body.cvat_task_id
    if body.cvat_job_id is not None:
        dataset.cvat_job_id = body.cvat_job_id
    if body.cvat_token is not None:
        dataset.cvat_token = body.cvat_token
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

def _apply_issue_resolution(db: Session, sample: Sample, issue: QCIssue, status: str, reviewer_note: str = ""):
    """Apply 1-Click resolution action to an issue and update ground truth annotation accordingly."""
    if status in ('ACCEPTED', 'RESOLVED'):
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
    elif status == 'PENDING':
        # Undo: remove added annotations or restore previous label
        for ann in list(sample.annotations):
            attrs = ann.attributes or {}
            if attrs.get('resolved_from_issue_id') == issue.id:
                db.delete(ann)
            elif attrs.get('previous_label') and issue.annotation_id == ann.id:
                ann.label = attrs['previous_label']
                attrs.pop('previous_label', None)
                attrs.pop('qc_verified', None)

    issue.status = status
    issue.reviewer_note = reviewer_note or ''

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
    _apply_issue_resolution(db, sample, issue, body.status, body.reviewer_note or "")

    db.commit()
    invalidate_qc_report_cache(sample.dataset_id)

    return {
        'id': issue.id,
        'status': issue.status,
        'reviewer_note': issue.reviewer_note,
        'issue_type': issue.issue_type,
        'suggested_label': issue.suggested_label
    }

@router.post('/samples/{sample_id}/qc/batch-resolve')
def batch_resolve_sample_issues_endpoint(
    sample_id: int,
    body: QCBatchResolveRequest,
    db: Session = Depends(get_db)
):
    sample = require(db, Sample, sample_id)
    pending_issues = [i for i in sample.qc_issues if i.status == 'PENDING']

    if body.issue_ids:
        targets = [i for i in pending_issues if i.id in body.issue_ids]
    elif body.consensus_tier:
        targets = [
            i for i in pending_issues
            if (i.evidence or {}).get('consensus_tier') == body.consensus_tier
        ]
    else:
        targets = pending_issues

    resolved_ids = []
    for issue in targets:
        _apply_issue_resolution(db, sample, issue, body.status, body.reviewer_note or "")
        resolved_ids.append(issue.id)

    db.commit()
    invalidate_qc_report_cache(sample.dataset_id)

    return {
        'success': True,
        'sample_id': sample_id,
        'resolved_count': len(resolved_ids),
        'resolved_issue_ids': resolved_ids
    }

@router.post('/datasets/{dataset_id}/qc/batch-resolve')
def batch_resolve_dataset_issues_endpoint(
    dataset_id: int,
    body: QCBatchResolveRequest,
    db: Session = Depends(get_db)
):
    dataset = require(db, Dataset, dataset_id)
    total_resolved = 0
    resolved_ids = []

    for sample in dataset.samples:
        pending_issues = [i for i in sample.qc_issues if i.status == 'PENDING']
        if body.consensus_tier:
            targets = [
                i for i in pending_issues
                if (i.evidence or {}).get('consensus_tier') == body.consensus_tier
            ]
        else:
            targets = pending_issues

        for issue in targets:
            _apply_issue_resolution(db, sample, issue, body.status, body.reviewer_note or "")
            resolved_ids.append(issue.id)
            total_resolved += 1

    db.commit()
    invalidate_qc_report_cache(dataset_id)

    return {
        'success': True,
        'dataset_id': dataset_id,
        'resolved_count': total_resolved,
        'resolved_issue_ids': resolved_ids
    }
