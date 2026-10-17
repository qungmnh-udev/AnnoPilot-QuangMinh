"""Validated training snapshots, persistent jobs, and checkpoint lifecycle."""
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import timezone
import random
import subprocess
import sys
import time
import uuid

from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from app.database import ROOT, UPLOADS
from app.models import Dataset, GoldenBenchmark, ModelCheckpoint, Sample, TrainingJob

CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle', 'traffic light', 'traffic sign']
MODELS = {'yolo26n': 'YOLO26n · Nhanh, dễ bắt đầu', 'yolo26s': 'YOLO26s · Chi tiết hơn',
          'rtdetr_l': 'RT-DETR · Transformer', 'fasterrcnn_mobilenet_v3': 'Faster R-CNN · Hai giai đoạn'}
ACTIVE = ('QUEUED', 'PREPARING', 'TRAINING', 'VALIDATING', 'REGISTERING')


def training_root():
    return Path(os.getenv('TRAINING_DIR', str(UPLOADS.parent / 'training-jobs'))).resolve()


def job_dir(job_id):
    if str(uuid.UUID(job_id)) != job_id:
        raise ValueError('Invalid training job ID')
    return training_root() / job_id


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def hardware():
    try:
        import torch
        gpu = torch.cuda.is_available()
        return dict(gpu_available=gpu, gpu_name=torch.cuda.get_device_name(0) if gpu else None,
                    gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if gpu else None,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    message='Sẵn sàng dùng GPU NVIDIA.' if gpu else
                    'Hiện dùng CPU. Muốn dùng GPU NVIDIA, cần PyTorch CUDA và driver tương thích.')
    except ImportError:
        return dict(gpu_available=False, gpu_name=None, gpu_memory_gb=None,
                    torch_version=None, cuda_version=None, message='Chưa cài thư viện model. Xem hướng dẫn cài đặt training.')


def select_device(requested):
    from app.services.yolo_detector import inference_device
    return inference_device('cuda:0' if requested == 'gpu' else requested)


def collect_snapshot(db, dataset, include_empty=False):
    """Exclude Golden images, including byte-identical copies in other datasets."""
    from app.schemas.qc import QCConfig
    from app.services.qc_engine import normalize_label
    golden_names = {}
    for benchmark in db.scalars(select(GoldenBenchmark)):
        golden_names.setdefault(benchmark.dataset_id, set()).update(f['file_name'] for f in benchmark.samples)
    golden_hashes = set()
    if golden_names:
        for sample in db.scalars(select(Sample).where(Sample.dataset_id.in_(golden_names))):
            if sample.file_name in golden_names[sample.dataset_id] and sample.media_path and Path(sample.media_path).is_file():
                golden_hashes.add(file_hash(sample.media_path))
    samples = db.scalars(select(Sample).where(Sample.dataset_id == dataset.id).order_by(Sample.id)
                         .options(selectinload(Sample.annotations))).all()
    counts = dict(total=len(samples), golden=0, empty=0, missing_media=0, duplicates=0)
    frames, seen = [], {}
    mapping = QCConfig().taxonomy_mapping
    for sample in samples:
        if sample.file_name in golden_names.get(dataset.id, set()):
            counts['golden'] += 1
            continue
        if not sample.media_path or not Path(sample.media_path).is_file():
            counts['missing_media'] += 1
            continue
        digest = file_hash(sample.media_path)
        if digest in golden_hashes:
            counts['golden'] += 1
            continue
        if not sample.annotations and not include_empty:
            counts['empty'] += 1
            continue
        with Image.open(sample.media_path) as image:
            width, height = image.size
        boxes = []
        for annotation in sample.annotations:
            if annotation.shape_type != 'BBOX_2D':
                raise ValueError(f'{sample.file_name}: chỉ hỗ trợ nhãn bounding box 2D để huấn luyện.')
            label = normalize_label(annotation.label, mapping)
            if label not in CLASSES:
                raise ValueError(f"{sample.file_name}: class '{annotation.label}' chưa thuộc bộ nhãn hỗ trợ.")
            g = annotation.geometry or {}
            if (set(g) != {'x1', 'y1', 'x2', 'y2'} or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in g.values()) or
                    not (0 <= g['x1'] < g['x2'] <= width and 0 <= g['y1'] < g['y2'] <= height)):
                raise ValueError(f'{sample.file_name}: box không hợp lệ hoặc vượt kích thước ảnh.')
            boxes.append(dict(label=label, geometry=g))
        boxes.sort(key=lambda b: (b['label'], tuple(b['geometry'][k] for k in ('x1', 'y1', 'x2', 'y2'))))
        if digest in seen:
            if seen[digest] != boxes:
                raise ValueError('Có ảnh trùng nội dung nhưng nhãn khác nhau. Hãy thống nhất nhãn trước khi train.')
            counts['duplicates'] += 1
            continue
        seen[digest] = boxes
        frames.append(dict(sample_id=sample.id, file_name=sample.file_name,
                           source_path=str(Path(sample.media_path).resolve()), sha256=digest,
                           width=width, height=height, annotations=boxes))
    counts['eligible'] = len(frames)
    return frames, counts


def split_frames(frames, fraction, seed):
    if len(frames) < 4:
        raise ValueError('Cần ít nhất 4 ảnh khác nhau có nhãn hợp lệ, ngoài tập Golden, để chia train và kiểm tra.')
    if not any(f['annotations'] for f in frames):
        raise ValueError('Cần ảnh có vật thể đã gán nhãn; không thể train chỉ bằng ảnh rỗng.')
    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    count = max(1, round(len(shuffled) * fraction))
    train, val = shuffled[count:], shuffled[:count]
    # Ensure the training partition contains objects without duplicating an image.
    if not any(f['annotations'] for f in train):
        annotated = next(i for i, f in enumerate(val) if f['annotations'])
        train[0], val[annotated] = val[annotated], train[0]
    return train, val


def job_view(db, job):
    db.refresh(job)
    progress = job.progress or {}
    heartbeat = progress.get('updated_at', job.created_at.replace(tzinfo=timezone.utc).timestamp())
    if job.status in ACTIVE and time.time() - heartbeat > 120:
        # Workers heartbeat even during long batches. A dead worker never holds the slot forever.
        job.status, job.active_slot = 'INTERRUPTED', None
        job.progress = {**progress, 'message': 'Tác vụ đã gián đoạn. Model QC trước đó được giữ; bạn có thể bắt đầu lượt mới.'}
        progress = job.progress
        db.commit()
    status = job.status if job.status not in ACTIVE else progress.get('status', job.status)
    return dict(id=job.id, dataset_id=job.dataset_id, model_name=job.model_name,
                status=status, config=job.config, created_at=job.created_at.isoformat(),
                **{k: v for k, v in progress.items() if k not in ('status', 'config', 'id', 'dataset_id', 'model_name', 'created_at')})


def launch_worker(job_id):
    directory = job_dir(job_id)
    env = os.environ.copy()
    env['TRAINING_DIR'] = str(training_root())
    env['PYTHONPATH'] = str(ROOT / 'backend')
    env['YOLO_CONFIG_DIR'] = str(directory / 'ultralytics-settings')
    env['MPLCONFIGDIR'] = str(directory / 'matplotlib')
    (directory / 'ultralytics-settings').mkdir(exist_ok=True)
    (directory / 'matplotlib').mkdir(exist_ok=True)
    env['PYTHONUTF8'] = '1'
    with (directory / 'training.log').open('w', encoding='utf-8') as log:
        return subprocess.Popen([sys.executable, '-m', 'app.services.training_worker', job_id],
                                cwd=ROOT / 'backend', env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def start_training(db, dataset, request):
    for active in db.scalars(select(TrainingJob).where(TrainingJob.active_slot.is_not(None))).all():
        job_view(db, active)
    if db.scalar(select(TrainingJob.id).where(TrainingJob.active_slot.is_not(None))):
        raise RuntimeError('Đang có một tác vụ huấn luyện. Chờ hoàn tất hoặc dừng trước khi chạy tiếp.')
    device = select_device(request.device)
    frames, counts = collect_snapshot(db, dataset, request.include_verified_empty)
    train, val = split_frames(frames, request.validation_fraction, request.seed)
    job_id = str(uuid.uuid4())
    config = {**request.model_dump(), 'resolved_device': device, 'counts': counts,
              'train_count': len(train), 'validation_count': len(val)}
    job = TrainingJob(id=job_id, dataset_id=dataset.id, model_name=request.model_name,
                      status='QUEUED', active_slot='training', config=config,
                      progress=dict(epoch=0, epochs=request.epochs, updated_at=time.time(), metrics={},
                                    message='Đang chuẩn bị dữ liệu huấn luyện.'))
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError('Một tác vụ huấn luyện khác vừa bắt đầu. Hãy chờ hoàn tất.')
    try:
        atomic_json(job_dir(job_id) / 'manifest.json', dict(job_id=job_id, dataset_id=dataset.id,
                    config=config, classes=CLASSES, train=train, val=val))
        launch_worker(job_id)
    except Exception as exc:
        job.status, job.active_slot = 'FAILED', None
        job.progress = {**(job.progress or {}), 'error': str(exc), 'message': 'Không khởi động được tiến trình huấn luyện.'}
        db.commit()
        raise
    return job_view(db, job)


def activate_checkpoint(db, dataset_id, checkpoint_id=None, model_name=None, commit=True):
    row = db.get(ModelCheckpoint, checkpoint_id) if checkpoint_id is not None else None
    if checkpoint_id is not None and (row is None or row.dataset_id != dataset_id):
        raise ValueError('Checkpoint không thuộc dataset này.')
    model_name = row.model_name if row else model_name
    if model_name not in MODELS:
        raise ValueError('Model không được hỗ trợ.')
    if row and (not Path(row.path).is_file() or file_hash(row.path) != row.sha256):
        raise ValueError('Checkpoint bị thiếu hoặc nội dung đã thay đổi; chưa thể dùng cho QC.')
    db.execute(update(ModelCheckpoint).where(ModelCheckpoint.dataset_id == dataset_id,
                ModelCheckpoint.model_name == model_name).values(active=False))
    if row:
        row.active = True
    # Existing predictions belong to the old version and must be explicitly rerun.
    for sample in db.scalars(select(Sample).where(Sample.dataset_id == dataset_id)):
        state = dict(sample.inference_state or {})
        state[model_name] = {**state.get(model_name, {}), 'active': True, 'status': 'STALE'}
        sample.inference_state = state
    if commit:
        db.commit()
    from app.services.qc_engine import invalidate_qc_report_cache
    invalidate_qc_report_cache(dataset_id)
    return dict(model_name=model_name, checkpoint_id=row.id if row else None, active=True)
