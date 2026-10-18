"""One isolated training process. API reloads do not interrupt its training loop."""
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time

from PIL import Image
from app.database import SessionLocal
from app.models import ModelCheckpoint, TrainingJob
from app.services.training import (activate_checkpoint, atomic_json, file_hash, job_dir)


class TrainingCancelled(Exception):
    pass


class Progress:
    def __init__(self, directory, epochs):
        self.directory = directory
        self.state = dict(status='PREPARING', epoch=0, epochs=epochs, metrics={},
                          message='Đang tạo bản sao dữ liệu cho lượt huấn luyện này.')
        self.lock = threading.RLock()
        self.done = threading.Event()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state['updated_at'] = time.time()
            # SQLite transactions avoid Windows file-sharing races between polling and writes.
            with SessionLocal() as db:
                job = db.get(TrainingJob, self.directory.name)
                if job is not None:
                    job.progress = dict(self.state)
                    job.status = self.state['status']
                    db.commit()

    def check(self, *_):
        if (self.directory / 'cancel').exists():
            raise TrainingCancelled()

    def heartbeat(self):
        while not self.done.wait(3):
            self.update()


def prepare_data(manifest, directory, progress):
    """Copy immutable inputs and export normalized labels; no image enters both splits."""
    for split in ('train', 'val'):
        for frame in manifest[split]:
            progress.check()
            source = Path(frame['source_path'])
            if not source.is_file() or file_hash(source) != frame['sha256']:
                raise ValueError(f"Ảnh {frame['file_name']} đã đổi sau khi bắt đầu. Hãy tạo lượt train mới.")
            image_path = directory / 'data' / 'images' / split / f"{frame['sha256']}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.convert('RGB').save(image_path)
            frame['image_path'] = str(image_path)
            labels = []
            for annotation in frame['annotations']:
                g, w, h = annotation['geometry'], frame['width'], frame['height']
                labels.append(f"{manifest['classes'].index(annotation['label'])} "
                              f"{(g['x1']+g['x2'])/(2*w):.9f} {(g['y1']+g['y2'])/(2*h):.9f} "
                              f"{(g['x2']-g['x1'])/w:.9f} {(g['y2']-g['y1'])/h:.9f}")
            label_path = directory / 'data' / 'labels' / split / (image_path.stem + '.txt')
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(labels), encoding='utf-8')
    # JSON is a YAML subset, accepted by Ultralytics without fragile path escaping.
    atomic_json(directory / 'data.yaml', dict(path=str(directory / 'data'), train='images/train',
                val='images/val', names=dict(enumerate(manifest['classes']))))
    atomic_json(directory / 'snapshot.json', manifest)


def finite_metrics(values):
    output = {}
    for key, value in values.items():
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            output[key] = value
    return output


def train_ultralytics(manifest, directory, progress):
    from ultralytics import YOLO, RTDETR
    from app.services.yolo_detector import get_yolo_weights_path
    from app.services.triad_detector import get_rtdetr_weights_path
    config = manifest['config']
    name = config['model_name']
    weights = get_rtdetr_weights_path() if name == 'rtdetr_l' else get_yolo_weights_path(name)
    if not weights or not Path(weights).is_file():
        raise RuntimeError('Thiếu checkpoint gốc của model. Cài weights trước khi huấn luyện.')
    model = (RTDETR if name == 'rtdetr_l' else YOLO)(str(weights))

    def epoch_end(trainer):
        progress.check()
        progress.update(status='TRAINING', epoch=trainer.epoch + 1,
                        metrics=finite_metrics(trainer.metrics), message='Đang học và kiểm tra trên tập validation.')

    for event in ('on_train_batch_start', 'on_val_batch_start', 'on_train_epoch_start'):
        model.add_callback(event, progress.check)
    model.add_callback('on_fit_epoch_end', epoch_end)
    progress.update(status='TRAINING', message='Đang khởi tạo model. Lượt đầu có thể mất một lúc.')
    model.train(data=str(directory / 'data.yaml'), epochs=config['epochs'], imgsz=config['image_size'],
                batch=config['batch_size'], device=config['resolved_device'], workers=0,
                seed=config['seed'], project=str(directory), name='fit', exist_ok=False,
                plots=False, cache=False, amp=config['resolved_device'].startswith('cuda'),
                patience=15, save=True, verbose=False)
    progress.check()
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training không tạo được best checkpoint; model QC cũ được giữ lại.')
    target = directory / 'best.pt'
    shutil.copy2(best, target)
    # Reload the actual artifact before activating it, not just the in-memory trainer model.
    validated = (RTDETR if name == 'rtdetr_l' else YOLO)(str(target))
    validated.predict(manifest['val'][0]['image_path'], imgsz=config['image_size'],
                      device=config['resolved_device'], verbose=False, save=False)
    return target, finite_metrics(model.trainer.metrics)


def detection_model(classes, image_size, pretrained=True):
    import torchvision.models.detection as detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None, min_size=image_size, max_size=image_size * 2)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(classes) + 1)
    return model


def train_fasterrcnn(manifest, directory, progress):
    import torch
    from torchvision.transforms.functional import to_tensor
    config, classes = manifest['config'], manifest['classes']
    device = torch.device(config['resolved_device'])
    torch.manual_seed(config['seed'])
    model = detection_model(classes, config['image_size']).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001, momentum=.9, weight_decay=.0005)
    best_loss = float('inf')
    target_path = directory / 'best.pth'

    def batch(frames):
        images, targets = [], []
        for frame in frames:
            with Image.open(frame['image_path']) as image:
                images.append(to_tensor(image.convert('RGB')).to(device))
            boxes = [[a['geometry'][k] for k in ('x1', 'y1', 'x2', 'y2')] for a in frame['annotations']]
            targets.append(dict(boxes=torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
                                labels=torch.tensor([classes.index(a['label']) + 1 for a in frame['annotations']], dtype=torch.int64, device=device)))
        return images, targets

    progress.update(status='TRAINING', message='Đang huấn luyện Faster R-CNN.')
    for epoch in range(config['epochs']):
        progress.check()
        model.train()
        shuffled = list(manifest['train'])
        __import__('random').Random(config['seed'] + epoch).shuffle(shuffled)
        train_loss, batches = 0., 0
        for index in range(0, len(shuffled), config['batch_size']):
            progress.check()
            images, targets = batch(shuffled[index:index + config['batch_size']])
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError('Loss không hữu hạn; kiểm tra nhãn và cấu hình training.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        progress.update(status='VALIDATING', message='Đang kiểm tra loss trên các ảnh validation riêng.')
        # TorchVision returns detection losses in train mode; freeze statistics and gradients.
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)):
                module.eval()
        val_loss = 0.
        with torch.no_grad():
            for frame in manifest['val']:
                progress.check()
                images, targets = batch([frame])
                loss = sum(model(images, targets).values()).item()
                if not math.isfinite(loss):
                    raise RuntimeError('Validation loss không hữu hạn.')
                val_loss += loss
        val_loss /= len(manifest['val'])
        metrics = dict(train_loss=train_loss / batches, validation_loss=val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(dict(state_dict=model.state_dict(), classes=classes, image_size=config['image_size']), target_path)
        progress.update(status='TRAINING', epoch=epoch + 1, metrics=metrics,
                        message='Đã hoàn thành một vòng học. Validation loss thấp hơn thường tốt hơn.')
    progress.check()
    # Verify a fresh instance can consume the exported checkpoint.
    from app.services.triad_detector import run_fasterrcnn_inference
    run_fasterrcnn_inference(manifest['val'][0]['image_path'], weights_path=target_path, device=config['resolved_device'])
    return target_path, dict(validation_loss=best_loss)


def run_job(job_id):
    directory = job_dir(job_id)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    progress = Progress(directory, manifest['config']['epochs'])
    progress.update()
    heartbeat = threading.Thread(target=progress.heartbeat, daemon=True)
    heartbeat.start()
    status = 'FAILED'
    try:
        import torch
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        progress.check()
        prepare_data(manifest, directory, progress)
        runner = train_fasterrcnn if manifest['config']['model_name'] == 'fasterrcnn_mobilenet_v3' else train_ultralytics
        checkpoint, metrics = runner(manifest, directory, progress)
        progress.check()
        progress.update(status='REGISTERING', message='Đang đưa checkpoint mới vào QC của dataset này.')
        digest = file_hash(checkpoint)
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if not job or job.active_slot != 'training':
                raise RuntimeError('Tác vụ không còn giữ quyền đăng ký checkpoint.')
            row = ModelCheckpoint(dataset_id=job.dataset_id, job_id=job_id, model_name=job.model_name,
                                  path=str(checkpoint), sha256=digest, metrics=metrics, active=False,
                                  training_images=[f['sha256'] for f in manifest['train'] + manifest['val']])
            db.add(row)
            db.flush()
            activate_checkpoint(db, job.dataset_id, checkpoint_id=row.id, commit=False)
            status = 'COMPLETED'
            job.status, job.active_slot = status, None
            db.commit()
            progress.update(status=status, checkpoint_id=row.id, metrics=metrics,
                            message='Huấn luyện xong. QC sẽ tự dùng model mới ở lần chạy tiếp theo.')
    except TrainingCancelled:
        status = 'CANCELLED'
        progress.update(status=status, message='Đã dừng huấn luyện. Giữ nguyên checkpoint QC trước đó.')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        message = str(exc)
        if 'out of memory' in message.lower():
            message = 'Không đủ bộ nhớ. Chọn YOLO26n, batch 1 hoặc giảm kích thước ảnh trong Tùy chọn thêm.'
        progress.update(status='FAILED', message='Huấn luyện chưa hoàn tất; checkpoint QC trước đó được giữ lại.', error=message)
    finally:
        progress.done.set()
        heartbeat.join(timeout=4)
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if job:
                job.status, job.active_slot = status, None
                db.commit()


if __name__ == '__main__':
    run_job(sys.argv[1])
