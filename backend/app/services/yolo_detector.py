"""YOLO26 checkpoint selection and shared .pt inference for queue and ensemble."""
import os
from pathlib import Path
import threading

MODEL_DIR = Path(__file__).resolve().parents[2] / 'models'
YOLO_MODELS = ('yolo26n', 'yolo26s')
DEFAULT_YOLO_MODEL = 'yolo26n'
_MODELS = {}
_LOCK = threading.RLock()


def resolve_yolo_model(model_name=None):
    name = model_name or os.getenv('ANNO_YOLO_MODEL', DEFAULT_YOLO_MODEL)
    if name not in YOLO_MODELS:
        raise ValueError('Choose yolo26n or yolo26s')
    return name


def get_yolo_weights_path(model_name=None):
    return MODEL_DIR / f'{resolve_yolo_model(model_name)}.pt'


def activate_yolo_state(previous, model, state):
    states = dict(previous or {})
    if model in YOLO_MODELS:
        for name in list(states):
            if name in YOLO_MODELS or name.startswith('yolov8'):
                states[name] = {**states[name], 'active': False}
        state = {**state, 'active': True}
    states[model] = state
    return states


def inference_device(device=None):
    import torch
    requested = device or os.getenv('ANNO_INFERENCE_DEVICE', 'auto')
    if requested == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested == 'cpu':
        return requested
    if requested in ('cuda', 'cuda:0', '0') and torch.cuda.is_available():
        return 'cuda:0'
    raise ValueError(f'Device {requested} is unavailable in the installed PyTorch environment')


def load_yolo(model_name=None, weights_path=None):
    config_dir = Path(os.getenv('YOLO_CONFIG_DIR', str(MODEL_DIR.parent / '.runtime' / 'ultralytics')))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config_dir))
    from ultralytics import YOLO
    path = Path(weights_path) if weights_path else get_yolo_weights_path(model_name)
    if not path.is_file():
        raise RuntimeError(f'YOLO checkpoint is unavailable: {path}. Download the official checkpoint into backend/models.')
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = YOLO(str(path))
        return _MODELS[key]


def predict_yolo(image_path, conf_thresh=.35, model_name=None, weights_path=None, device=None, imgsz=640):
    selected = resolve_yolo_model(model_name)
    from app.services.triad_detector import COCO_TO_BDD100K, BDD100K_CLASSES
    with _LOCK:  # Ultralytics Predictor is shared; serialize queue/on-demand access.
        model = load_yolo(selected, weights_path)
        results = model.predict(source=str(image_path), conf=conf_thresh, imgsz=imgsz,
                                device=inference_device(device), verbose=False, save=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                raw = model.names[int(box.cls[0])].lower()
                label = COCO_TO_BDD100K.get(raw, raw)
                if label not in BDD100K_CLASSES:
                    continue
                boxes.append({'label': label, 'confidence': round(float(box.conf[0]), 3),
                              'geometry': dict(zip(('x1', 'y1', 'x2', 'y2'), [round(v, 2) for v in box.xyxy[0].tolist()])),
                              'source_model': selected})
        return boxes
