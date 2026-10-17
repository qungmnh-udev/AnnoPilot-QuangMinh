from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Dataset, ModelCheckpoint, TrainingJob
from app.schemas.training import TrainingRequest, TrainModel
from app.services.training import (ACTIVE, MODELS, activate_checkpoint, collect_snapshot, hardware,
                                   job_dir, job_view, start_training)

router = APIRouter(prefix='/api')
_checkpoint_versions = {}


def dataset_or_404(db, dataset_id):
    row = db.get(Dataset, dataset_id)
    if row is None:
        raise HTTPException(404, 'Dataset not found')
    return row


def job_or_404(db, dataset_id, job_id):
    row = db.get(TrainingJob, job_id)
    if row is None or row.dataset_id != dataset_id:
        raise HTTPException(404, 'Training job not found')
    return row


@router.get('/training/hardware')
def training_hardware():
    return hardware()


@router.get('/datasets/{dataset_id}/training/preview')
def training_preview(dataset_id: int, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    try:
        _, counts = collect_snapshot(db, dataset)
        return dict(**counts, ready=counts['eligible'] >= 4, models=MODELS)
    except (ValueError, OSError) as exc:
        return dict(ready=False, error=str(exc), models=MODELS)


@router.get('/datasets/{dataset_id}/training')
def training_history(dataset_id: int, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    jobs = db.scalars(select(TrainingJob).where(TrainingJob.dataset_id == dataset_id)
                      .order_by(TrainingJob.created_at.desc()).limit(20)).all()
    checkpoints = db.scalars(select(ModelCheckpoint).where(ModelCheckpoint.dataset_id == dataset_id)
                             .order_by(ModelCheckpoint.id.desc())).all()
    # Report cache is process-local, whereas training may finish in another process.
    version = tuple(c.id for c in checkpoints if c.active)
    if _checkpoint_versions.get(dataset_id) != version:
        from app.services.qc_engine import invalidate_qc_report_cache
        invalidate_qc_report_cache(dataset_id)
        _checkpoint_versions[dataset_id] = version
    return dict(jobs=[job_view(db, j) for j in jobs], checkpoints=[
        dict(id=c.id, model_name=c.model_name, job_id=c.job_id, active=c.active,
             metrics=c.metrics, sha256=c.sha256, created_at=c.created_at.isoformat()) for c in checkpoints])


@router.post('/datasets/{dataset_id}/training', status_code=202)
def train(dataset_id: int, request: TrainingRequest, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    try:
        return start_training(db, dataset, request)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@router.post('/datasets/{dataset_id}/training/{job_id}/cancel')
def cancel(dataset_id: int, job_id: str, db: Session = Depends(get_db)):
    job = job_or_404(db, dataset_id, job_id)
    state = job_view(db, job)
    if state['status'] in ACTIVE:
        (job_dir(job_id) / 'cancel').touch()
    return dict(status=state['status'], message='Đã yêu cầu dừng. Model cũ tiếp tục được giữ cho đến khi có checkpoint hoàn tất.')


@router.get('/datasets/{dataset_id}/training/{job_id}/log')
def log(dataset_id: int, job_id: str, db: Session = Depends(get_db)):
    job_or_404(db, dataset_id, job_id)
    path = job_dir(job_id) / 'training.log'
    if not path.is_file():
        raise HTTPException(404, 'Log chưa sẵn sàng')
    return FileResponse(path, media_type='text/plain', filename=f'training-{job_id}.log')


class ActivationRequest(BaseModel):
    model_name: TrainModel
    checkpoint_id: int | None = None


@router.post('/datasets/{dataset_id}/training/checkpoint')
def select_checkpoint(dataset_id: int, request: ActivationRequest, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    row = db.get(ModelCheckpoint, request.checkpoint_id) if request.checkpoint_id is not None else None
    if row and row.model_name != request.model_name:
        raise HTTPException(422, 'Checkpoint không cùng loại model đã chọn.')
    try:
        return activate_checkpoint(db, dataset_id, request.checkpoint_id, request.model_name)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
