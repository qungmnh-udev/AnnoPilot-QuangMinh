"""
Triad Multi-Model Detector Service (Option B: YOLO26n + Faster R-CNN + RT-DETR)
Provides unified inference, class taxonomy normalization (COCO -> BDD100K),
and dataset prediction generation for architectural ensemble diversity.

Architectural Triad:
  1. M1 (YOLO26n): 1-Stage CNN (Anchor-free, Task-Aligned Assigner)
  2. M2 (Faster R-CNN MobileNetV3): 2-Stage CNN (Anchor-based, RPN + RoIAlign)
  3. M3 (RT-DETR-l): Vision Transformer (Hybrid Encoder + Deformable Attention Decoder)
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from time import perf_counter
from datetime import datetime, timezone

from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import delete

from app.models import Dataset, Sample, Prediction
from app.schemas.qc import QCConfig
from app.services.yolo_detector import (get_yolo_weights_path, predict_yolo, inference_device, activate_yolo_state)

logger = logging.getLogger(__name__)

# COCO 80-class to BDD100K 10-class taxonomy mapping
COCO_TO_BDD100K: Dict[str, str] = {
    'person': 'pedestrian',
    'rider': 'rider',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'train': 'train',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    'traffic light': 'traffic light',
    'stop sign': 'traffic sign',
}

BDD100K_CLASSES = {
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
}

TRIAD_SPECS: Dict[str, Dict[str, Any]] = {
    "yolo26n": {
        "label": "YOLO26n",
        "family": "CNN 1-Stage",
        "architecture": "YOLO26 detection with dual heads",
        "default_weight": 1.0,
        "description": "Siêu nhanh, cân bằng tốt giữa tốc độ và độ nhạy cho các khung hình liên tục."
    },
    "yolo26s": {
        "label": "YOLO26s",
        "family": "CNN 1-Stage",
        "architecture": "YOLO26 small detection with dual heads",
        "default_weight": 1.0,
        "description": "Biến thể small, chọn thay YOLO26n; đo tốc độ và chất lượng trên dataset thực."
    },
    "fasterrcnn_mobilenet_v3": {
        "label": "Faster R-CNN MobileNetV3",
        "family": "CNN 2-Stage",
        "architecture": "RPN + RoIAlign + MobileNetV3 FPN",
        "default_weight": 1.0,
        "description": "Hai giai đoạn, ít False Positive, ranh giới hộp viền localization chặt chẽ."
    },
    "rtdetr_l": {
        "label": "RT-DETR-l",
        "family": "Vision Transformer",
        "architecture": "Hybrid Encoder + Deformable Attention Decoder",
        "default_weight": 1.2,
        "description": "Self-attention toàn cục, bắt vật thể che khuất và đông đúc vượt trội."
    }
}

_MODEL_CACHE: Dict[str, Any] = {}


def get_rtdetr_weights_path() -> Optional[Path]:
    """Find local RT-DETR weights."""
    candidates = [
        Path("models/rtdetr-l.pt"),
        Path("backend/models/rtdetr-l.pt"),
        Path("rtdetr-l.pt"),
        Path("backend/rtdetr-l.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def run_yolo_inference(image_path, conf_thresh=.35, weights_path=None, model_name='yolo26n'):
    return predict_yolo(image_path, conf_thresh, model_name, weights_path)


def run_fasterrcnn_inference(
    image_path: str,
    conf_thresh: float = 0.35
) -> List[Dict[str, Any]]:
    """Execute Faster R-CNN MobileNetV3 inference with COCO -> BDD100K normalization."""
    try:
        import torch
        import torchvision.models.detection as d
        import torchvision.transforms.functional as F

        cache_key = "fasterrcnn_mobilenet_v3"
        if cache_key not in _MODEL_CACHE:
            weights = d.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = d.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            model.eval()
            _MODEL_CACHE[cache_key] = (model, weights.meta["categories"])

        model, categories = _MODEL_CACHE[cache_key]

        img = Image.open(image_path).convert("RGB")
        img_t = F.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_t)[0]

        boxes = []
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        bbox_coords = preds["boxes"].cpu().numpy()

        for score, lbl_idx, box in zip(scores, labels, bbox_coords):
            if score < conf_thresh:
                continue
            cat_name = categories[lbl_idx].lower() if lbl_idx < len(categories) else "unknown"
            normalized_label = COCO_TO_BDD100K.get(cat_name)
            if not normalized_label or normalized_label not in BDD100K_CLASSES:
                continue

            boxes.append({
                "label": normalized_label,
                "confidence": round(float(score), 3),
                "geometry": {
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2)
                },
                "source_model": "fasterrcnn_mobilenet_v3"
            })
        return boxes
    except Exception as e:
        raise RuntimeError(f"Faster R-CNN inference failed: {e}") from e


def run_rtdetr_inference(
    image_path: str,
    conf_thresh: float = 0.35,
    weights_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute RT-DETR Transformer inference with COCO -> BDD100K normalization."""
    try:
        from ultralytics import RTDETR
        w_path = weights_path or (str(get_rtdetr_weights_path()) if get_rtdetr_weights_path() else "rtdetr-l.pt")
        cache_key = f"rtdetr_{w_path}"
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = RTDETR(w_path)
        model = _MODEL_CACHE[cache_key]

        results = model.predict(source=image_path, conf=conf_thresh, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                raw_label = model.names.get(cls_id, f"class_{cls_id}").lower()
                normalized_label = COCO_TO_BDD100K.get(raw_label, raw_label)
                if normalized_label not in BDD100K_CLASSES:
                    continue

                boxes.append({
                    "label": normalized_label,
                    "confidence": round(conf, 3),
                    "geometry": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    },
                    "source_model": "rtdetr_l"
                })
        return boxes
    except Exception as e:
        raise RuntimeError(f"RT-DETR inference failed: {e}") from e


def detect_triad_frame(image_path, conf_thresh=.35, models=None, timings=None):
    selected_models = models or ['yolo26n', 'fasterrcnn_mobilenet_v3', 'rtdetr_l']
    if len(set(selected_models)) != len(selected_models) or any(m not in TRIAD_SPECS for m in selected_models):
        raise ValueError('Select distinct supported detector models')
    if 'yolo26n' in selected_models and 'yolo26s' in selected_models:
        raise ValueError('Choose one YOLO26 variant per ensemble')
    runners = {'yolo26n': run_yolo_inference, 'yolo26s': lambda path, **kw: run_yolo_inference(path, model_name='yolo26s', **kw), 'fasterrcnn_mobilenet_v3': run_fasterrcnn_inference,
               'rtdetr_l': run_rtdetr_inference}
    results = {}
    for model in selected_models:
        started = perf_counter()
        results[model] = runners[model](image_path, conf_thresh=conf_thresh)
        if timings is not None:
            timings[model] = perf_counter() - started
    return results


def run_dataset_triad_inference(
    db: Session,
    dataset_id: int,
    models: Optional[List[str]] = None,
    conf_thresh: float = 0.35,
    sample_limit: Optional[int] = None,
    auto_ensemble: bool = True,
    weights: Optional[Dict[str, float]] = None,
    iou_thresh: float = 0.55
) -> Dict[str, Any]:
    """
    Run Option B Triad across all samples in the dataset:
      1. Generates predictions from each selected model
      2. Persists them into `Prediction` table with appropriate source_model
      3. (Optional) Immediately triggers Tri-Tier WBF ensemble
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset #{dataset_id} not found")

    selected_models = models or ["yolo26n", "fasterrcnn_mobilenet_v3", "rtdetr_l"]
    if len(set(selected_models)) != len(selected_models) or any(m not in TRIAD_SPECS for m in selected_models):
        raise ValueError('Select distinct supported detector models')
    if 'yolo26n' in selected_models and 'yolo26s' in selected_models:
        raise ValueError('Choose one YOLO26 variant per ensemble')
    default_weights = {"yolo26n": 1.0, "yolo26s": 1.0, "fasterrcnn_mobilenet_v3": 1.0, "rtdetr_l": 1.2}
    if weights:
        default_weights.update(weights)

    samples = dataset.samples
    if sample_limit and sample_limit > 0:
        samples = samples[:sample_limit]

    from app.services.ensemble import DetectionBox, weighted_box_fusion
    from app.services.qc_engine import audit_single_frame, invalidate_qc_report_cache
    model_counts = {m: 0 for m in selected_models}
    frames_processed = 0
    skipped_frames = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    total_fused = 0
    for sample in samples:
        if not sample.media_path or not os.path.isfile(sample.media_path):
            skipped_frames.append(sample.file_name)
            sample.inference_state = {**(sample.inference_state or {}), **{m: {'status': 'ERROR', 'error': 'Media unavailable'} for m in selected_models}}
            db.commit()
            invalidate_qc_report_cache(dataset_id)
            continue
        timings = {}
        try:
            frame_results = detect_triad_frame(sample.media_path, conf_thresh=conf_thresh, models=selected_models, timings=timings)
        except Exception as exc:
            sample.inference_state = {**(sample.inference_state or {}), **{m: {'status': 'ERROR', 'error': str(exc)} for m in selected_models}}
            db.commit()
            invalidate_qc_report_cache(dataset_id)
            raise
        db.execute(delete(Prediction).where(Prediction.sample_id == sample.id,
                   Prediction.source_model.in_(selected_models)))
        db.execute(delete(Prediction).where(Prediction.sample_id == sample.id,
                   Prediction.source_model.startswith('ensemble_')))
        state = dict(sample.inference_state or {})
        for model_name, boxes in frame_results.items():
            checkpoint = str(get_yolo_weights_path(model_name)) if model_name in ('yolo26n', 'yolo26s') else (str(get_rtdetr_weights_path()) if model_name == 'rtdetr_l' else 'torchvision/FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT')
            from app.services.provenance import detector_provenance
            provenance = {**detector_provenance(checkpoint), 'conf_thresh': conf_thresh, 'inference_seconds': timings.get(model_name),
                          'completed_at': datetime.now(timezone.utc).isoformat()}
            state = activate_yolo_state(state, model_name, {'status': 'SUCCESS', 'device': inference_device() if model_name.startswith('yolo26') else 'cpu/default', 'imgsz': 640 if model_name.startswith('yolo26') else None, **provenance})
            for box in boxes:
                db.add(Prediction(sample_id=sample.id, label=box['label'], geometry=box['geometry'],
                                  confidence=box['confidence'], source_model=model_name, extra_data=provenance))
                model_counts[model_name] += 1
        sample.inference_state = state
        if auto_ensemble:
            per_model = {m: [DetectionBox(label=b['label'], confidence=b['confidence'],
                         **b['geometry'], source_model=m) for b in boxes] for m, boxes in frame_results.items()}
            for fb in weighted_box_fusion(per_model, model_weights=default_weights, iou_threshold=iou_thresh, confidence_threshold=conf_thresh):
                db.add(Prediction(sample_id=sample.id, label=fb.label, geometry=fb.geometry, confidence=fb.confidence,
                                  source_model='ensemble_wbf_v1', extra_data={**fb.evidence, 'tier': fb.tier}))
                tier_counts[fb.tier] += 1
                total_fused += 1
        db.flush()
        db.expire(sample, ['predictions'])
        audit_single_frame(db, sample.id, run_detector=False, config=QCConfig(expected_models=selected_models, model_weights=default_weights))
        frames_processed += 1  # result is visible after every committed frame
    ensemble_res = None if not auto_ensemble else {
        'status': 'success', 'total_fused_predictions': total_fused,
        'models_used': selected_models, 'files_processed': frames_processed,
        'tier_breakdown': {'tier_1_high_consensus': tier_counts[1], 'tier_2_class_disagreement': tier_counts[2], 'tier_3_model_divergence': tier_counts[3]},
    }
    return {
        "success": not skipped_frames,
        "skipped_frames": skipped_frames,
        "frames_processed": frames_processed,
        "selected_models": selected_models,
        "model_counts": model_counts,
        "total_predictions": sum(model_counts.values()),
        "ensemble_result": ensemble_res
    }
