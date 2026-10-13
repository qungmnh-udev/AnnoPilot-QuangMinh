"""
Triad Multi-Model Detector Service (Option B: YOLOv8n + Faster R-CNN + RT-DETR)
Provides unified inference, class taxonomy normalization (COCO -> BDD100K),
and dataset prediction generation for architectural ensemble diversity.

Architectural Triad:
  1. M1 (YOLOv8n): 1-Stage CNN (Anchor-free, Task-Aligned Assigner)
  2. M2 (Faster R-CNN MobileNetV3): 2-Stage CNN (Anchor-based, RPN + RoIAlign)
  3. M3 (RT-DETR-l): Vision Transformer (Hybrid Encoder + Deformable Attention Decoder)
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import torch
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import delete

from app.models import Dataset, Sample, Prediction
from app.services.ensemble import run_ensemble_from_existing_predictions

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
    "yolov8n": {
        "label": "YOLOv8n",
        "family": "CNN 1-Stage",
        "architecture": "Anchor-free CSPDarknet + TAL",
        "default_weight": 1.0,
        "description": "Siêu nhanh, cân bằng tốt giữa tốc độ và độ nhạy cho các khung hình liên tục."
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


def get_yolo_weights_path() -> Optional[Path]:
    """Find local YOLO weights."""
    candidates = [
        Path("models/yolov8n_bdd100k.pt"),
        Path("backend/models/yolov8n_bdd100k.pt"),
        Path("../training/runs/qc_finetuned/weights/best.pt"),
        Path("../training/yolov8n_bdd100k_init.pt"),
        Path("models/yolov8n.pt"),
        Path("backend/models/yolov8n.pt"),
        Path("yolov8n.pt"),
        Path("backend/yolov8n.pt"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


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


def run_yolo_inference(
    image_path: str,
    conf_thresh: float = 0.35,
    weights_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute YOLOv8 inference."""
    try:
        from ultralytics import YOLO
        w_path = weights_path or (str(get_yolo_weights_path()) if get_yolo_weights_path() else "yolov8n.pt")
        cache_key = f"yolo_{w_path}"
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = YOLO(w_path)
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
                    "source_model": "yolov8n"
                })
        return boxes
    except Exception as e:
        logger.error(f"Error running YOLOv8 inference on {image_path}: {e}")
        return []


def run_fasterrcnn_inference(
    image_path: str,
    conf_thresh: float = 0.35
) -> List[Dict[str, Any]]:
    """Execute Faster R-CNN MobileNetV3 inference with COCO -> BDD100K normalization."""
    try:
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
        logger.error(f"Error running Faster R-CNN inference on {image_path}: {e}")
        return []


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
        logger.error(f"Error running RT-DETR inference on {image_path}: {e}")
        return []


def detect_triad_frame(
    image_path: str,
    conf_thresh: float = 0.35,
    models: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run Option B inference on a single frame for requested models.
    Returns: Dict[model_name, list_of_boxes]
    """
    selected_models = models or ["yolov8n", "fasterrcnn_mobilenet_v3", "rtdetr_l"]
    results: Dict[str, List[Dict[str, Any]]] = {}

    if "yolov8n" in selected_models:
        results["yolov8n"] = run_yolo_inference(image_path, conf_thresh=conf_thresh)

    if "fasterrcnn_mobilenet_v3" in selected_models:
        results["fasterrcnn_mobilenet_v3"] = run_fasterrcnn_inference(image_path, conf_thresh=conf_thresh)

    if "rtdetr_l" in selected_models:
        results["rtdetr_l"] = run_rtdetr_inference(image_path, conf_thresh=conf_thresh)

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

    selected_models = models or ["yolov8n", "fasterrcnn_mobilenet_v3", "rtdetr_l"]
    default_weights = {"yolov8n": 1.0, "fasterrcnn_mobilenet_v3": 1.0, "rtdetr_l": 1.2}
    if weights:
        default_weights.update(weights)

    samples = dataset.samples
    if sample_limit and sample_limit > 0:
        samples = samples[:sample_limit]

    # Pre-clean existing predictions from these models to prevent duplicate appending
    for m in selected_models:
        db.execute(
            delete(Prediction).where(
                Prediction.sample_id.in_([s.id for s in samples]),
                Prediction.source_model == m
            )
        )
    db.commit()

    model_counts: Dict[str, int] = {m: 0 for m in selected_models}
    frames_processed = 0

    for sample in samples:
        if not sample.media_path or not os.path.isfile(sample.media_path):
            continue

        frame_results = detect_triad_frame(
            sample.media_path,
            conf_thresh=conf_thresh,
            models=selected_models
        )
        frames_processed += 1

        for model_name, boxes in frame_results.items():
            for b in boxes:
                pred = Prediction(
                    sample_id=sample.id,
                    label=b["label"],
                    geometry=b["geometry"],
                    confidence=b["confidence"],
                    source_model=model_name
                )
                db.add(pred)
                model_counts[model_name] += 1

    db.commit()

    ensemble_res = None
    if auto_ensemble and sum(model_counts.values()) > 0:
        ensemble_res = run_ensemble_from_existing_predictions(
            db=db,
            dataset_id=dataset_id,
            model_names=selected_models,
            model_weights=default_weights,
            iou_threshold=iou_thresh,
            confidence_threshold=conf_thresh,
            ensemble_source_model="ensemble_wbf_v1",
            run_audit=True
        )

    return {
        "success": True,
        "frames_processed": frames_processed,
        "selected_models": selected_models,
        "model_counts": model_counts,
        "total_predictions": sum(model_counts.values()),
        "ensemble_result": ensemble_res
    }
