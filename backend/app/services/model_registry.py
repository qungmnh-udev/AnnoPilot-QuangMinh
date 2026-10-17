"""Dataset-scoped checkpoints; original weights remain available as a rollback."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from sqlalchemy import inspect, select
from app.database import SessionLocal, engine
from app.models import ModelCheckpoint

_dataset = ContextVar('checkpoint_dataset', default=None)


@contextmanager
def checkpoint_scope(dataset_id):
    token = _dataset.set(dataset_id)
    try:
        yield
    finally:
        _dataset.reset(token)


def active_checkpoint(model_name, dataset_id=None):
    dataset_id = dataset_id if dataset_id is not None else _dataset.get()
    if dataset_id is None or not inspect(engine).has_table('model_checkpoints'):
        return None
    with SessionLocal() as db:
        row = db.scalar(select(ModelCheckpoint).where(
            ModelCheckpoint.dataset_id == dataset_id,
            ModelCheckpoint.model_name == model_name, ModelCheckpoint.active.is_(True)
        ).order_by(ModelCheckpoint.id.desc()))
        if row is None:
            return None
        path = Path(row.path)
        if not path.is_file():
            raise RuntimeError('Checkpoint đang dùng cho QC bị thiếu. Chọn lại model gốc trong mục Huấn luyện.')
        return dict(id=row.id, job_id=row.job_id, path=str(path), sha256=row.sha256,
                    model_name=row.model_name, metrics=row.metrics)


def checkpoint_identity_matches(state, model_name, dataset_id):
    current = active_checkpoint(model_name, dataset_id)
    expected = current['sha256'] if current else None
    return state.get('checkpoint_sha256') == expected if current else not state.get('training_job_id')
