"""Read-only dataset snapshots and isolated, asynchronous real-model experiments."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import ROOT, get_db
from app.models import Dataset, GoldenBenchmark
from app.schemas.experiments import WBFExperimentRequest
from app.services.wbf_experiments import run_experiment

router = APIRouter(prefix='/api/datasets/{dataset_id}/ensemble/experiments')
REPORT_ROOT = ROOT / '.runtime' / 'wbf-experiments'
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='wbf-experiment')
_lock = threading.RLock()
_active = {}


def dataset_or_404(db, dataset_id):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, 'Dataset not found')
    return dataset


def report_path(dataset_id, run_id):
    try:
        if str(uuid.UUID(run_id)) != run_id:
            raise ValueError()
    except ValueError:
        raise HTTPException(404, 'Experiment not found')
    return REPORT_ROOT / str(dataset_id) / f'{run_id}.json'


def view_report(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    run_id = path.stem
    with _lock:
        active = _active.get(run_id)
        if active:
            data.update(active)
        elif data['status'] == 'RUNNING':
            data['status'] = 'INTERRUPTED'
    data.pop('frames', None)  # Raw predictions are available through the report download.
    return dict(id=run_id, **data)


@router.get('')
def list_experiments(dataset_id: int, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    paths = sorted((REPORT_ROOT / str(dataset_id)).glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return [view_report(path) for path in paths[:20]]


@router.post('', status_code=202)
def start_experiment(dataset_id: int, request: WBFExperimentRequest, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    golden = None
    if request.benchmark_id is not None:
        benchmark = db.get(GoldenBenchmark, request.benchmark_id)
        if not benchmark or benchmark.dataset_id != dataset_id:
            raise HTTPException(422, 'Golden benchmark does not belong to this dataset')
        golden = {frame['file_name']: frame['annotations'] for frame in benchmark.samples}
    samples = sorted(dataset.samples, key=lambda s: s.id)
    if golden is not None:
        samples = [s for s in samples if s.file_name in golden]
    samples = samples[:request.sample_limit]
    if not samples or any(not s.media_path or not Path(s.media_path).is_file() for s in samples):
        raise HTTPException(422, 'Every selected frame needs an existing image; upload matching media first')
    if golden is not None:
        from sqlalchemy import select
        from app.models import ModelCheckpoint
        from app.services.training import file_hash
        trained = db.scalars(select(ModelCheckpoint).where(ModelCheckpoint.dataset_id == dataset_id,
                             ModelCheckpoint.active.is_(True), ModelCheckpoint.model_name.in_(request.models))).all()
        used = {digest for checkpoint in trained for digest in checkpoint.training_images or []}
        if used and any(file_hash(s.media_path) in used for s in samples):
            raise HTTPException(422, 'Golden images were used to train/validate an active checkpoint. Select independent images or restore the original model.')
    frames = [dict(file_name=s.file_name, image_path=s.media_path, dataset_id=dataset_id,
                   annotations=[dict(id=a.id, shape_type=a.shape_type, label=a.label, geometry=a.geometry) for a in s.annotations],
                   golden=golden[s.file_name] if golden is not None else None) for s in samples]
    run_id = str(uuid.uuid4())
    path = report_path(dataset_id, run_id)
    with _lock:
        if _active:
            raise HTTPException(409, 'A real-model experiment is already running; wait for it to finish')
        _active[run_id] = dict(status='RUNNING', total_frames=len(frames), processed_frames=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        from app.services.wbf_experiments import write_report
        write_report(path, dict(status='RUNNING', config=request.model_dump(), total_frames=len(frames),
                               processed_frames=0, summary={}, created_at=datetime.now(timezone.utc).isoformat()))

    def work():
        try:
            def progress(state):
                with _lock:
                    _active[run_id].update(state)
            run_experiment(frames, request, path, progress=progress)
        finally:
            with _lock:
                _active.pop(run_id, None)
    _executor.submit(work)
    return dict(id=run_id, status='RUNNING', total_frames=len(frames))


@router.get('/{run_id}')
def get_experiment(dataset_id: int, run_id: str, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    path = report_path(dataset_id, run_id)
    if not path.is_file():
        raise HTTPException(404, 'Experiment not found')
    return view_report(path)


@router.get('/{run_id}/report')
def download_experiment(dataset_id: int, run_id: str, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    path = report_path(dataset_id, run_id)
    if not path.is_file():
        raise HTTPException(404, 'Experiment not found')
    return FileResponse(path, media_type='application/json', filename=f'wbf-experiment-{run_id}.json')
