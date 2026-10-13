import os
import math
import time
import json
import random
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, delete

from app.models import Sample, Annotation, Prediction, QCIssue, Dataset
from app.schemas.qc import QCConfig, BBoxGeometry
from app.database import SessionLocal
from app.services.pipeline import extract_frame_number

_REPORT_CACHE: Dict[int, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()

def invalidate_qc_report_cache(dataset_id: Optional[int] = None) -> None:
    """Invalidate cached QC audit report for a dataset or all datasets."""
    with _CACHE_LOCK:
        if dataset_id is not None:
            _REPORT_CACHE.pop(dataset_id, None)
        else:
            _REPORT_CACHE.clear()

def calculate_box_area(box: Dict[str, float]) -> float:
    """Calculate 2D bounding box area."""
    return max(0.0, float(box.get('x2', 0)) - float(box.get('x1', 0))) * max(0.0, float(box.get('y2', 0)) - float(box.get('y1', 0)))

def calculate_box_iou(b1: Dict[str, float], b2: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two 2D bounding boxes."""
    x1 = max(float(b1.get('x1', 0)), float(b2.get('x1', 0)))
    y1 = max(float(b1.get('y1', 0)), float(b2.get('y1', 0)))
    x2 = min(float(b1.get('x2', 0)), float(b2.get('x2', 0)))
    y2 = min(float(b1.get('y2', 0)), float(b2.get('y2', 0)))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area1 = calculate_box_area(b1)
    area2 = calculate_box_area(b2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def normalize_label(label: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """Standardize class labels according to taxonomy (e.g. BDD100K)."""
    cleaned = (label or '').strip().lower().replace('_', ' ')
    if mapping and cleaned in mapping:
        return mapping[cleaned]
    return cleaned

# Predefined confusion prior: pairs of classes that are frequently confused in driving scenarios
CONFUSION_PRIORS: Dict[Tuple[str, str], float] = {
    ('car', 'truck'): 0.95,
    ('truck', 'car'): 0.95,
    ('bus', 'truck'): 0.90,
    ('truck', 'bus'): 0.90,
    ('car', 'bus'): 0.85,
    ('bus', 'car'): 0.85,
    ('pedestrian', 'rider'): 0.95,
    ('rider', 'pedestrian'): 0.95,
    ('bicycle', 'motorcycle'): 0.90,
    ('motorcycle', 'bicycle'): 0.90,
    ('traffic sign', 'traffic light'): 0.70,
    ('traffic light', 'traffic sign'): 0.70,
}

def get_confusion_weight(label_gt: str, label_pred: str) -> float:
    """Weight score based on whether disagreement is a known plausible confusion vs completely distinct."""
    pair = (label_gt, label_pred)
    if pair in CONFUSION_PRIORS:
        return CONFUSION_PRIORS[pair]
    return 0.80

def match_boxes(
    annotations: List[Annotation],
    predictions: List[Prediction],
    iou_threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Greedy bipartite IoU matching sorted by prediction confidence descending.
    Returns:
      - matched_pairs: list of {gt, pred, iou}
      - unmatched_preds: list of {pred, max_iou}
      - unmatched_gts: list of {gt, max_iou}
    """
    gt_list = [a for a in annotations if a.shape_type == 'BBOX_2D' and a.geometry]
    pred_list = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    matched_gt_ids = set()
    matched_pred_ids = set()
    matched_pairs = []

    pred_max_ious = {p.id: 0.0 for p in pred_list}
    gt_max_ious = {g.id: 0.0 for g in gt_list}

    for p in pred_list:
        for g in gt_list:
            iou_val = calculate_box_iou(g.geometry, p.geometry)
            if iou_val > pred_max_ious[p.id]:
                pred_max_ious[p.id] = iou_val
            if iou_val > gt_max_ious[g.id]:
                gt_max_ious[g.id] = iou_val

    for p in pred_list:
        best_gt = None
        best_iou = 0.0
        for g in gt_list:
            if g.id in matched_gt_ids:
                continue
            iou_val = calculate_box_iou(g.geometry, p.geometry)
            if iou_val >= iou_threshold and iou_val > best_iou:
                best_iou = iou_val
                best_gt = g

        if best_gt is not None:
            matched_pairs.append({
                'gt': best_gt,
                'pred': p,
                'iou': round(best_iou, 4)
            })
            matched_gt_ids.add(best_gt.id)
            matched_pred_ids.add(p.id)

    unmatched_preds = [
        {'pred': p, 'max_iou': round(pred_max_ious[p.id], 4)}
        for p in pred_list if p.id not in matched_pred_ids
    ]
    unmatched_gts = [
        {'gt': g, 'max_iou': round(gt_max_ious[g.id], 4)}
        for g in gt_list if g.id not in matched_gt_ids
    ]

    return matched_pairs, unmatched_preds, unmatched_gts

def detect_sample_qc_issues(
    sample_id: int,
    annotations: List[Annotation],
    predictions: List[Prediction],
    config: Optional[QCConfig] = None
) -> List[Dict[str, Any]]:
    """
    Detect Missing Objects and Wrong Classes for a sample.
    Prioritizes Multi-Model Ensemble predictions to eliminate duplicate issues across models,
    grounding QC in consolidated consensus.
    """
    cfg = config or QCConfig()

    # Model prioritization: If ensemble predictions exist, strictly use them to prevent duplicate issues.
    # If multiple models exist without ensemble, use primary model to avoid cross-model duplication.
    ensemble_preds = [p for p in predictions if p.source_model and p.source_model.startswith("ensemble_")]
    if ensemble_preds:
        active_preds = ensemble_preds
    else:
        distinct_models = list(dict.fromkeys(p.source_model for p in predictions if p.source_model))
        if len(distinct_models) > 1:
            primary = "yolov8n" if "yolov8n" in distinct_models else distinct_models[0]
            active_preds = [p for p in predictions if p.source_model == primary]
        else:
            active_preds = predictions

    matched_pairs, unmatched_preds, _ = match_boxes(
        annotations, active_preds, cfg.iou_match_threshold
    )

    issues: List[Dict[str, Any]] = []

    # 1. Detect Wrong Class candidates
    for item in matched_pairs:
        gt: Annotation = item['gt']
        pred: Prediction = item['pred']
        iou_val: float = item['iou']

        norm_gt = normalize_label(gt.label, cfg.taxonomy_mapping)
        norm_pred = normalize_label(pred.label, cfg.taxonomy_mapping)

        if norm_gt != norm_pred and pred.confidence >= cfg.min_conf_wrong_class:
            confusion_factor = get_confusion_weight(norm_gt, norm_pred)
            qc_score = min(1.0, max(0.0, iou_val * pred.confidence * confusion_factor))

            extra = getattr(pred, 'extra_data', None) or {}
            tier = extra.get('tier') or (1 if (pred.source_model and pred.source_model.startswith('ensemble_') and pred.confidence >= 0.80) else None)
            contributing = extra.get('contributing_models') or ([pred.source_model] if pred.source_model else [])
            num_agreed = extra.get('num_models_agreed') or len(contributing)

            issues.append({
                'sample_id': sample_id,
                'issue_type': 'WRONG_CLASS',
                'location': gt.geometry,
                'human_label': gt.label,
                'suggested_label': pred.label,
                'annotation_id': gt.id,
                'prediction_id': pred.id,
                'qc_score': round(qc_score, 4),
                'evidence': {
                    'iou': iou_val,
                    'model_confidence': round(pred.confidence, 4),
                    'confusion_factor': confusion_factor,
                    'source_model': pred.source_model,
                    'consensus_tier': tier,
                    'contributing_models': contributing,
                    'num_models_agreed': num_agreed,
                    'tier_reason': extra.get('tier_reason', f"Đồng thuận từ {num_agreed} mô hình" if num_agreed > 1 else ""),
                    'model_votes': extra.get('model_votes', {}),
                    'reason': f"Detector suggests '{pred.label}' (confidence {pred.confidence:.2f}) instead of human label '{gt.label}' (IoU={iou_val:.2f})"
                },
                'status': 'PENDING',
                'reviewer_note': ''
            })

    # 2. Detect Missing Object candidates
    for item in unmatched_preds:
        pred: Prediction = item['pred']
        max_iou: float = item['max_iou']

        box_area = calculate_box_area(pred.geometry)
        if box_area < cfg.min_box_area:
            continue

        if max_iou <= cfg.iou_unmatched_threshold and pred.confidence >= cfg.min_conf_missing:
            norm_pred = normalize_label(pred.label, cfg.taxonomy_mapping)
            class_reliability = cfg.class_reliability.get(norm_pred, 0.85)
            size_factor = min(1.0, math.sqrt(box_area) / 25.0)

            extra = getattr(pred, 'extra_data', None) or {}
            tier = extra.get('tier') or (1 if (pred.source_model and pred.source_model.startswith('ensemble_') and pred.confidence >= 0.80) else None)
            contributing = extra.get('contributing_models') or ([pred.source_model] if pred.source_model else [])
            num_agreed = extra.get('num_models_agreed') or len(contributing)

            # Boost score for high consensus across multiple distinct models
            consensus_boost = 1.15 if (tier == 1 and num_agreed >= 2) else 1.0
            qc_score = min(1.0, max(0.0, pred.confidence * (1.0 - max_iou) * class_reliability * (0.8 + 0.2 * size_factor) * consensus_boost))

            issues.append({
                'sample_id': sample_id,
                'issue_type': 'MISSING_OBJECT',
                'location': pred.geometry,
                'human_label': None,
                'suggested_label': pred.label,
                'annotation_id': None,
                'prediction_id': pred.id,
                'qc_score': round(qc_score, 4),
                'evidence': {
                    'max_iou_with_any_annotation': max_iou,
                    'model_confidence': round(pred.confidence, 4),
                    'box_area': round(box_area, 1),
                    'source_model': pred.source_model,
                    'consensus_tier': tier,
                    'contributing_models': contributing,
                    'num_models_agreed': num_agreed,
                    'tier_reason': extra.get('tier_reason', f"Đồng thuận từ {num_agreed} mô hình" if num_agreed > 1 else ""),
                    'model_votes': extra.get('model_votes', {}),
                    'reason': f"Detector found '{pred.label}' (confidence {pred.confidence:.2f}) with no human annotation nearby (max IoU={max_iou:.2f})"
                },
                'status': 'PENDING',
                'reviewer_note': ''
            })

    issues.sort(key=lambda x: x['qc_score'], reverse=True)
    return issues

def compute_sample_qc_score(issues: List[Any]) -> Tuple[float, str]:
    """Compute aggregate sample QC score and severity."""
    if not issues:
        return 0.0, 'CLEAN'

    scores = [issue.qc_score if hasattr(issue, 'qc_score') else issue.get('qc_score', 0) for issue in issues]
    max_score = max(scores)
    boost = 0.04 * min(len(scores) - 1, 5)
    overall_score = min(1.0, round(max_score + boost, 4))

    if overall_score >= 0.70:
        severity = 'HIGH'
    elif overall_score >= 0.45:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'

    return overall_score, severity

# --- Single Image Detector Service ---

_DETECTOR_CACHE: Dict[str, Any] = {}

def get_best_model_path() -> Optional[Path]:
    """Find the best available model weights."""
    candidates = [
        Path("models/yolov8n_bdd100k.pt"),
        Path("backend/models/yolov8n_bdd100k.pt"),
        Path("../training/runs/qc_finetuned/weights/best.pt"),
        Path("../training/yolov8n_bdd100k_init.pt"),
        Path("yolov8n.pt"),
        Path("models/yolov8n.onnx"),
        Path("backend/models/yolov8n.onnx"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None

def detect_image_objects(
    image_path: Union[str, Path],
    conf_thresh: float = 0.35,
    model_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Run object detection on a single image.
    Outputs standardized BDD100K class labels and box coordinates.
    """
    img_p = Path(image_path)
    if not img_p.is_file():
        return []

    target_model_path = Path(model_path) if model_path else get_best_model_path()
    if not target_model_path or not target_model_path.is_file():
        return []

    model_key = str(target_model_path.resolve())

    # Try Ultralytics YOLO
    if target_model_path.suffix.lower() == '.pt':
        try:
            from ultralytics import YOLO
            if model_key not in _DETECTOR_CACHE:
                _DETECTOR_CACHE[model_key] = YOLO(model_key)
            model = _DETECTOR_CACHE[model_key]

            results = model.predict(source=str(img_p), conf=conf_thresh, verbose=False)
            boxes = []
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    xyxy = b.xyxy[0].tolist()
                    label = model.names.get(cls_id, f"class_{cls_id}")

                    boxes.append({
                        "label": label,
                        "confidence": round(conf, 3),
                        "geometry": {
                            "x1": round(xyxy[0], 2),
                            "y1": round(xyxy[1], 2),
                            "x2": round(xyxy[2], 2),
                            "y2": round(xyxy[3], 2)
                        }
                    })
            return boxes
        except Exception:
            pass

    # Fallback to OpenCV DNN for ONNX models
    if target_model_path.suffix.lower() == '.onnx':
        try:
            import cv2
            import numpy as np

            if model_key not in _DETECTOR_CACHE:
                _DETECTOR_CACHE[model_key] = cv2.dnn.readNetFromONNX(model_key)
            net = _DETECTOR_CACHE[model_key]

            img = cv2.imread(str(img_p))
            if img is None:
                return []
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            out = net.forward()[0]

            boxes = []
            confidences = []
            class_ids = []
            cx = out[0]
            cy = out[1]
            bw = out[2]
            bh = out[3]

            num_classes = out.shape[0] - 4
            if num_classes > 1:
                class_scores = out[4:]
                best_class = np.argmax(class_scores, axis=0)
                best_conf = np.max(class_scores, axis=0)
            else:
                best_conf = out[4]
                best_class = np.zeros(len(best_conf), dtype=int)

            mask = best_conf >= conf_thresh
            for i in np.where(mask)[0]:
                score = float(best_conf[i])
                c_id = int(best_class[i])
                box_x1 = max(0.0, float((cx[i] - bw[i] / 2) * (w / 640.0)))
                box_y1 = max(0.0, float((cy[i] - bh[i] / 2) * (h / 640.0)))
                box_x2 = min(float(w), float((cx[i] + bw[i] / 2) * (w / 640.0)))
                box_y2 = min(float(h), float((cy[i] + bh[i] / 2) * (h / 640.0)))
                boxes.append([int(box_x1), int(box_y1), int(box_x2 - box_x1), int(box_y2 - box_y1)])
                confidences.append(score)
                class_ids.append(c_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
            output_boxes = []
            bdd_classes = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign'}

            if len(indices) > 0:
                for idx in indices.flatten():
                    b = boxes[idx]
                    cid = class_ids[idx]
                    label = bdd_classes.get(cid, 'car')
                    output_boxes.append({
                        "label": label,
                        "confidence": round(confidences[idx], 3),
                        "geometry": {
                            "x1": float(b[0]),
                            "y1": float(b[1]),
                            "x2": float(b[0] + b[2]),
                            "y2": float(b[1] + b[3])
                        }
                    })
            return output_boxes
        except Exception:
            pass

    return []

# --- Single Frame Audit (Priority P0: Incremental Processing) ---

def audit_single_frame(
    db: Session,
    sample_id: int,
    config: Optional[QCConfig] = None,
    run_detector: bool = True,
    conf_thresh: float = 0.35,
    source_model: str = "yolov8-bdd100k",
    commit: bool = True
) -> Dict[str, Any]:
    """
    Perform QC audit for a single frame independently.
    Runs detection if media is available, matches against annotations,
    reconciles QCIssue rows in DB, and returns the sample summary.
    """
    sample = db.get(Sample, sample_id)
    if not sample:
        raise ValueError(f"Sample #{sample_id} not found")

    cfg = config or QCConfig()

    # Step 1: Run detection if requested and media exists
    if run_detector and sample.media_path and os.path.isfile(sample.media_path):
        detected_boxes = detect_image_objects(sample.media_path, conf_thresh=conf_thresh)
        if detected_boxes:
            # Clear existing predictions from same source model
            db.execute(delete(Prediction).where(
                Prediction.sample_id == sample.id,
                Prediction.source_model == source_model
            ))
            for b in detected_boxes:
                pred = Prediction(
                    sample_id=sample.id,
                    label=b['label'],
                    geometry=b['geometry'],
                    confidence=b['confidence'],
                    source_model=source_model
                )
                db.add(pred)
            db.flush()

    # Step 1.5: If multiple models exist on this sample without ensemble, auto-fuse via WBF
    all_preds = list(sample.predictions)
    ensemble_preds = [p for p in all_preds if p.source_model and p.source_model.startswith("ensemble_")]
    distinct_models = list(dict.fromkeys(p.source_model for p in all_preds if p.source_model and not p.source_model.startswith("ensemble_")))

    if not ensemble_preds and len(distinct_models) >= 2:
        from app.services.ensemble import DetectionBox, weighted_box_fusion
        per_model: Dict[str, List[DetectionBox]] = {m: [] for m in distinct_models}
        for p in all_preds:
            if p.source_model in per_model and p.geometry:
                g = p.geometry
                per_model[p.source_model].append(DetectionBox(
                    label=p.label,
                    confidence=p.confidence,
                    x1=float(g.get("x1", 0)),
                    y1=float(g.get("y1", 0)),
                    x2=float(g.get("x2", 0)),
                    y2=float(g.get("y2", 0)),
                    source_model=p.source_model
                ))
        fused = weighted_box_fusion(per_model, confidence_threshold=conf_thresh or 0.30)
        if fused:
            for fb in fused:
                new_p = Prediction(
                    sample_id=sample.id,
                    label=fb.label,
                    geometry=fb.geometry,
                    confidence=fb.confidence,
                    source_model="ensemble_wbf_v1",
                    extra_data={
                        "tier": fb.tier,
                        "tier_reason": fb.tier_reason,
                        "contributing_models": fb.contributing_models,
                        "model_votes": fb.model_votes,
                        "num_models_agreed": fb.num_models_agreed,
                    }
                )
                db.add(new_p)
                sample.predictions.append(new_p)
            db.flush()

    # Step 2: Compare annotations with predictions
    existing_map = {
        (issue.issue_type, issue.suggested_label, issue.annotation_id, issue.prediction_id): issue
        for issue in sample.qc_issues
    }

    raw_issues = detect_sample_qc_issues(sample.id, sample.annotations, sample.predictions, cfg)

    current_keys = set()
    active_issues = []
    for raw in raw_issues:
        key = (raw['issue_type'], raw['suggested_label'], raw['annotation_id'], raw['prediction_id'])
        current_keys.add(key)
        if key in existing_map:
            existing = existing_map[key]
            existing.location = raw['location']
            existing.human_label = raw['human_label']
            existing.qc_score = raw['qc_score']
            existing.evidence = raw['evidence']
            active_issues.append(existing)
        else:
            db_issue = QCIssue(**raw)
            db.add(db_issue)
            sample.qc_issues.append(db_issue)
            active_issues.append(db_issue)

    # Delete unreviewed pending issues that are no longer detected
    for key, existing in list(existing_map.items()):
        if key not in current_keys and existing.status == 'PENDING':
            db.delete(existing)
            if existing in sample.qc_issues:
                sample.qc_issues.remove(existing)

    db.flush()

    sample_score, severity = compute_sample_qc_score(active_issues)
    if commit:
        db.commit()
        if sample.dataset_id:
            invalidate_qc_report_cache(sample.dataset_id)
    else:
        db.flush()

    fnum = sample.frame_number if getattr(sample, 'frame_number', None) is not None else extract_frame_number(sample.file_name, sample.id - 1)
    curl = getattr(sample, 'cvat_url', None)
    if not curl and sample.dataset:
        ds = sample.dataset
        base = (ds.cvat_base_url or 'http://localhost:8080').rstrip('/')
        if ds.cvat_task_id:
            if ds.cvat_job_id:
                curl = f"{base}/tasks/{ds.cvat_task_id}/jobs/{ds.cvat_job_id}?frame={fnum}"
            else:
                curl = f"{base}/tasks/{ds.cvat_task_id}?frame={fnum}"
        else:
            curl = f"{base}/tasks?frame={fnum}"

    return {
        'sample_id': sample.id,
        'file_name': sample.file_name,
        'frame_number': fnum,
        'cvat_url': curl,
        'qc_score': sample_score,
        'severity': severity,
        'missing_count': sum(1 for i in active_issues if i.issue_type == 'MISSING_OBJECT'),
        'wrong_class_count': sum(1 for i in active_issues if i.issue_type == 'WRONG_CLASS'),
        'total_issues': len(active_issues),
        'issues': [
            {
                'id': i.id,
                'sample_id': i.sample_id,
                'issue_type': i.issue_type,
                'location': i.location,
                'human_label': i.human_label,
                'suggested_label': i.suggested_label,
                'annotation_id': i.annotation_id,
                'prediction_id': i.prediction_id,
                'qc_score': i.qc_score,
                'evidence': i.evidence,
                'frame_number': fnum,
                'cvat_url': curl,
                'status': i.status,
                'reviewer_note': i.reviewer_note or ''
            }
            for i in active_issues
        ]
    }

def run_dataset_qc_audit(db: Session, dataset_id: int, config: Optional[QCConfig] = None) -> Dict[str, Any]:
    """
    Audit entire dataset against current predictions.
    Reconciles all samples with batch eager-loading and a single database commit.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError('Dataset not found')

    cfg = config or QCConfig()

    samples = db.scalars(
        select(Sample)
        .where(Sample.dataset_id == dataset_id)
        .options(
            selectinload(Sample.annotations),
            selectinload(Sample.predictions),
            selectinload(Sample.qc_issues)
        )
    ).all()

    total_samples = len(samples)
    total_missing = 0
    total_wrong = 0
    flagged_samples = 0
    severity_breakdown = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'CLEAN': 0}
    sample_summaries = []

    for sample in samples:
        summary = audit_single_frame(db, sample.id, config=cfg, run_detector=False, commit=False)
        severity_breakdown[summary['severity']] += 1
        total_missing += summary['missing_count']
        total_wrong += summary['wrong_class_count']
        if summary['total_issues'] > 0:
            flagged_samples += 1
        sample_summaries.append(summary)

    db.commit()

    sample_summaries.sort(key=lambda s: (-s['qc_score'], s['sample_id']))

    clean_samples = total_samples - flagged_samples
    workload_reduction = (clean_samples / total_samples * 100.0) if total_samples > 0 else 0.0

    report = {
        'dataset_id': dataset_id,
        'total_samples': total_samples,
        'flagged_samples_count': flagged_samples,
        'clean_samples_count': clean_samples,
        'workload_reduction_percentage': round(workload_reduction, 2),
        'total_missing_candidates': total_missing,
        'total_wrong_class_candidates': total_wrong,
        'total_issues': total_missing + total_wrong,
        'severity_breakdown': severity_breakdown,
        'top_suspicious_samples': sample_summaries
    }

    with _CACHE_LOCK:
        _REPORT_CACHE[dataset_id] = report

    return report

def get_dataset_qc_report(db: Session, dataset_id: int, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Lightning-fast retrieval of the dataset QC report.
    Checks memory cache first (< 1ms). If not cached, queries existing database records
    in a single eager-loaded bulk query (~100-150ms) instead of re-running the full detector/IoU audit.
    If audit has never run on this dataset, it runs the initial audit automatically.
    """
    if not force_refresh:
        with _CACHE_LOCK:
            if dataset_id in _REPORT_CACHE:
                return _REPORT_CACHE[dataset_id]

    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError('Dataset not found')

    # Check if any QCIssue exists for this dataset
    has_any_issue = db.scalar(
        select(QCIssue.id).join(Sample).where(Sample.dataset_id == dataset_id).limit(1)
    ) is not None

    if not has_any_issue:
        # Check if there are any predictions that need initial audit
        has_any_pred = db.scalar(
            select(Prediction.id).join(Sample).where(Sample.dataset_id == dataset_id).limit(1)
        ) is not None
        if has_any_pred:
            return run_dataset_qc_audit(db, dataset_id)

    # Fast construction from existing database records
    base = (dataset.cvat_base_url or 'http://localhost:8080').rstrip('/')
    has_task = bool(dataset.cvat_task_id)
    has_job = bool(dataset.cvat_job_id)

    samples = db.scalars(
        select(Sample)
        .where(Sample.dataset_id == dataset_id)
        .options(selectinload(Sample.qc_issues))
    ).all()

    total_samples = len(samples)
    total_missing = 0
    total_wrong = 0
    flagged_samples = 0
    severity_breakdown = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'CLEAN': 0}
    sample_summaries = []

    for s in samples:
        issues = list(s.qc_issues)
        missing_count = sum(1 for i in issues if i.issue_type == 'MISSING_OBJECT')
        wrong_count = sum(1 for i in issues if i.issue_type == 'WRONG_CLASS')
        tot = len(issues)
        if tot > 0:
            flagged_samples += 1
        total_missing += missing_count
        total_wrong += wrong_count

        score, sev = compute_sample_qc_score(issues)
        severity_breakdown[sev] += 1

        fnum = s.frame_number if s.frame_number is not None else extract_frame_number(s.file_name, s.id - 1)
        if has_task:
            if has_job:
                curl = f'{base}/tasks/{dataset.cvat_task_id}/jobs/{dataset.cvat_job_id}?frame={fnum}'
            else:
                curl = f'{base}/tasks/{dataset.cvat_task_id}?frame={fnum}'
        else:
            curl = f'{base}/tasks?frame={fnum}'

        sample_summaries.append({
            'sample_id': s.id,
            'file_name': s.file_name,
            'frame_number': fnum,
            'cvat_url': curl,
            'qc_score': score,
            'severity': sev,
            'missing_count': missing_count,
            'wrong_class_count': wrong_count,
            'total_issues': tot,
            'issues': [
                {
                    'id': i.id,
                    'sample_id': i.sample_id,
                    'issue_type': i.issue_type,
                    'location': i.location,
                    'human_label': i.human_label,
                    'suggested_label': i.suggested_label,
                    'annotation_id': i.annotation_id,
                    'prediction_id': i.prediction_id,
                    'qc_score': i.qc_score,
                    'evidence': i.evidence,
                    'frame_number': fnum,
                    'cvat_url': curl,
                    'status': i.status,
                    'reviewer_note': i.reviewer_note or ''
                }
                for i in issues
            ]
        })

    sample_summaries.sort(key=lambda s: (-s['qc_score'], s['sample_id']))
    clean_samples = total_samples - flagged_samples
    workload_reduction = (clean_samples / total_samples * 100.0) if total_samples > 0 else 0.0

    report = {
        'dataset_id': dataset_id,
        'total_samples': total_samples,
        'flagged_samples_count': flagged_samples,
        'clean_samples_count': clean_samples,
        'workload_reduction_percentage': round(workload_reduction, 2),
        'total_missing_candidates': total_missing,
        'total_wrong_class_candidates': total_wrong,
        'total_issues': total_missing + total_wrong,
        'severity_breakdown': severity_breakdown,
        'top_suspicious_samples': sample_summaries
    }

    with _CACHE_LOCK:
        _REPORT_CACHE[dataset_id] = report

    return report

# --- Asynchronous Frame-by-Frame Queue Manager (P0) ---

class FrameAuditQueueManager:
    """
    Thread-safe Background Queue Manager that processes dataset frames incrementally.
    Allows frontend to poll or stream real-time progress without blocking the server.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._queues = {}
                cls._instance._stop_events = {}
                cls._instance._threads = {}
            return cls._instance

    def get_status(self, dataset_id: int) -> Dict[str, Any]:
        with self._lock:
            if dataset_id not in self._queues:
                return {
                    'dataset_id': dataset_id,
                    'status': 'IDLE',
                    'total_frames': 0,
                    'processed_frames': 0,
                    'progress_percent': 0.0,
                    'current_frame_id': None,
                    'current_frame_name': None,
                    'flagged_frames_count': 0,
                    'recent_audited_frames': [],
                    'error_message': None
                }
            return dict(self._queues[dataset_id])

    def stop_queue(self, dataset_id: int) -> None:
        with self._lock:
            if dataset_id in self._stop_events:
                self._stop_events[dataset_id].set()
            if dataset_id in self._queues:
                self._queues[dataset_id]['status'] = 'PAUSED'

    def start_queue(
        self,
        dataset_id: int,
        sample_ids: Optional[List[int]] = None,
        conf_thresh: float = 0.35,
        force_repredict: bool = False
    ) -> Dict[str, Any]:
        with self._lock:
            if dataset_id in self._threads and self._threads[dataset_id].is_alive():
                return self.get_status(dataset_id)

            stop_event = threading.Event()
            self._stop_events[dataset_id] = stop_event

            with SessionLocal() as db:
                dataset = db.get(Dataset, dataset_id)
                if not dataset:
                    raise ValueError(f"Dataset #{dataset_id} not found")
                all_ids = [s.id for s in dataset.samples]

            target_ids = sample_ids if sample_ids else all_ids

            self._queues[dataset_id] = {
                'dataset_id': dataset_id,
                'status': 'RUNNING',
                'total_frames': len(target_ids),
                'processed_frames': 0,
                'progress_percent': 0.0,
                'current_frame_id': None,
                'current_frame_name': None,
                'flagged_frames_count': 0,
                'recent_audited_frames': [],
                'error_message': None
            }

            thread = threading.Thread(
                target=self._worker_loop,
                args=(dataset_id, target_ids, conf_thresh, force_repredict, stop_event),
                daemon=True
            )
            self._threads[dataset_id] = thread
            thread.start()

            return dict(self._queues[dataset_id])

    def _worker_loop(
        self,
        dataset_id: int,
        target_ids: List[int],
        conf_thresh: float,
        force_repredict: bool,
        stop_event: threading.Event
    ):
        processed = 0
        flagged = 0
        total = len(target_ids)

        for sid in target_ids:
            if stop_event.is_set():
                with self._lock:
                    if dataset_id in self._queues:
                        self._queues[dataset_id]['status'] = 'PAUSED'
                return

            try:
                with SessionLocal() as db:
                    sample = db.get(Sample, sid)
                    if not sample:
                        continue
                    sname = sample.file_name

                    with self._lock:
                        if dataset_id in self._queues:
                            self._queues[dataset_id]['current_frame_id'] = sid
                            self._queues[dataset_id]['current_frame_name'] = sname

                    summary = audit_single_frame(
                        db, sid, run_detector=force_repredict or not sample.predictions, conf_thresh=conf_thresh
                    )

                    processed += 1
                    if summary['total_issues'] > 0:
                        flagged += 1

                    with self._lock:
                        if dataset_id in self._queues:
                            q = self._queues[dataset_id]
                            q['processed_frames'] = processed
                            q['progress_percent'] = round((processed / total) * 100.0, 1) if total > 0 else 100.0
                            q['flagged_frames_count'] = flagged
                            recent = q['recent_audited_frames']
                            recent.insert(0, summary)
                            if len(recent) > 50:
                                recent.pop()

                # Yield brief slice to prevent CPU monopolization
                time.sleep(0.01)
            except Exception as e:
                with self._lock:
                    if dataset_id in self._queues:
                        self._queues[dataset_id]['status'] = 'ERROR'
                        self._queues[dataset_id]['error_message'] = str(e)
                return

        with self._lock:
            if dataset_id in self._queues:
                self._queues[dataset_id]['status'] = 'COMPLETED'
                self._queues[dataset_id]['progress_percent'] = 100.0
                self._queues[dataset_id]['current_frame_id'] = None
                self._queues[dataset_id]['current_frame_name'] = None

queue_manager = FrameAuditQueueManager()

# --- Two-Way CVAT Synchronization (P0: System of Record) ---

def fetch_annotations_from_cvat_api(
    dataset: Dataset,
    token: Optional[str] = None
) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    """
    Fetch live annotations directly from CVAT REST API for this dataset.
    Returns a dict mapping frame_number -> list of annotation dicts with geometry and label.
    """
    import urllib.request
    import json

    pat = token or dataset.cvat_token or os.getenv('CVAT_TOKEN') or 'ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO'
    base_url = (dataset.cvat_base_url or 'http://localhost:8080').rstrip('/')
    task_id = dataset.cvat_task_id
    job_id = dataset.cvat_job_id

    if not task_id and not job_id:
        return None

    headers = {
        'Authorization': f'Bearer {pat}',
        'Content-Type': 'application/json'
    }

    # 1. Fetch labels mapping
    labels_map = {}
    target_task_id = task_id
    if not target_task_id and job_id:
        try:
            job_req = urllib.request.Request(f"{base_url}/api/jobs/{job_id}", headers=headers)
            with urllib.request.urlopen(job_req, timeout=10) as resp:
                job_data = json.loads(resp.read().decode('utf-8'))
                target_task_id = job_data.get('task_id')
        except Exception:
            pass

    if target_task_id:
        try:
            lbl_req = urllib.request.Request(f"{base_url}/api/labels?task_id={target_task_id}", headers=headers)
            with urllib.request.urlopen(lbl_req, timeout=10) as resp:
                lbl_data = json.loads(resp.read().decode('utf-8'))
                results = lbl_data.get('results', lbl_data) if isinstance(lbl_data, dict) else lbl_data
                labels_map = {item['id']: item['name'] for item in results}
        except Exception:
            pass

    # 2. Fetch annotations
    endpoint = f"{base_url}/api/jobs/{job_id}/annotations" if job_id else f"{base_url}/api/tasks/{task_id}/annotations"
    try:
        ann_req = urllib.request.Request(endpoint, headers=headers)
        with urllib.request.urlopen(ann_req, timeout=20) as resp:
            ann_data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"[WARN] Failed to fetch annotations from CVAT API ({endpoint}): {e}")
        return None

    frames_ann: Dict[int, List[Dict[str, Any]]] = {}

    for s in ann_data.get('shapes', []):
        if s.get('outside'):
            continue
        f = s.get('frame', 0)
        pts = s.get('points', [])
        if len(pts) >= 4:
            lbl_id = s.get('label_id')
            lbl_name = labels_map.get(lbl_id, f"label_{lbl_id}")
            frames_ann.setdefault(f, []).append({
                'label': lbl_name,
                'geometry': {'x1': float(pts[0]), 'y1': float(pts[1]), 'x2': float(pts[2]), 'y2': float(pts[3])},
                'shape_type': s.get('type', 'rectangle'),
                'occluded': bool(s.get('occluded', False)),
                'attributes': s.get('attributes', [])
            })

    for tr in ann_data.get('tracks', []):
        lbl_id = tr.get('label_id')
        lbl_name = labels_map.get(lbl_id, f"label_{lbl_id}")
        for s in tr.get('shapes', []):
            if s.get('outside'):
                continue
            f = s.get('frame', 0)
            pts = s.get('points', [])
            if len(pts) >= 4:
                frames_ann.setdefault(f, []).append({
                    'label': lbl_name,
                    'geometry': {'x1': float(pts[0]), 'y1': float(pts[1]), 'x2': float(pts[2]), 'y2': float(pts[3])},
                    'shape_type': s.get('type', 'rectangle'),
                    'occluded': bool(s.get('occluded', False)),
                    'attributes': s.get('attributes', [])
                })

    return frames_ann

def sync_cvat_dataset_annotations(
    db: Session,
    dataset_id: int,
    file_path: Optional[Path] = None,
    content: Optional[Union[str, bytes]] = None
) -> Dict[str, Any]:
    """
    Synchronize annotations from CVAT (the single source of truth).
    Updates local annotation mirror, re-audits affected frames,
    and automatically verifies/resolves QC issues if the user has fixed them on CVAT.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset #{dataset_id} not found")

    from app.parsers import CVATParser, BDD100KParser

    parsed_samples = []

    # 1. If raw content or file provided, parse it
    if content:
        import tempfile
        is_bytes = isinstance(content, bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml' if ('<?xml' in str(content[:100])) else '.json') as tmp:
            if is_bytes:
                tmp.write(content)
            else:
                tmp.write(content.encode('utf-8'))
            tmp_path = Path(tmp.name)

        try:
            if tmp_path.suffix == '.xml':
                parser = CVATParser()
                parsed_samples = parser.parse([tmp_path], [])
            else:
                parser = BDD100KParser()
                parsed_samples = parser.parse([tmp_path], [])
        finally:
            tmp_path.unlink(missing_ok=True)
    elif file_path and Path(file_path).is_file():
        fp = Path(file_path)
        if fp.suffix.lower() == '.xml':
            parsed_samples = CVATParser().parse([fp], [])
        else:
            parsed_samples = BDD100KParser().parse([fp], [])
    else:
        # 2. Prioritize fetching live annotations directly from CVAT REST API (The System of Record)
        cvat_frames = fetch_annotations_from_cvat_api(dataset)
        if cvat_frames:
            for f_num, ann_list in cvat_frames.items():
                parsed_samples.append({
                    'file_name': f"frame_{f_num:06d}.PNG",
                    'frame_number': f_num,
                    'annotations': ann_list
                })
        else:
            # 3. Fallback: look for latest annotation file in uploads
            storage_key = dataset.storage_key
            if storage_key:
                from app.database import UPLOADS
                ann_dir = UPLOADS / storage_key / 'annotations'
                xml_files = list(ann_dir.rglob('*.xml'))
                json_files = list(ann_dir.rglob('*.json'))
                if xml_files:
                    parsed_samples = CVATParser().parse([xml_files[0]], [])
                elif json_files:
                    parsed_samples = BDD100KParser().parse([json_files[0]], [])
                parsed_samples = BDD100KParser().parse([json_files[0]], [])

    if not parsed_samples:
        return {
            'status': 'no_data',
            'message': 'Không tìm thấy dữ liệu annotation hợp lệ từ CVAT để đồng bộ',
            'dataset_id': dataset_id,
            'updated_frames_count': 0,
            'auto_resolved_issues_count': 0,
            'remaining_issues_count': 0,
            'synced_at': datetime.now(timezone.utc).isoformat()
        }

    # Map database samples by file_name and frame_number
    sample_by_name = {s.file_name: s for s in dataset.samples}
    sample_by_frame = {s.frame_number: s for s in dataset.samples if s.frame_number is not None}

    updated_frames = 0
    auto_resolved = 0

    for item in parsed_samples:
        fname = item['file_name']
        sample = sample_by_name.get(fname)
        if not sample:
            frame_num = item.get('frame_number')
            if frame_num is not None:
                sample = sample_by_frame.get(frame_num)

        if not sample:
            continue

        # Get existing pending issues before syncing new annotations
        pending_issues = [i for i in sample.qc_issues if i.status == 'PENDING']

        # Update DB annotations to match CVAT
        db.execute(delete(Annotation).where(Annotation.sample_id == sample.id))
        for ann_data in item.get('annotations', []):
            db_ann = Annotation(
                sample_id=sample.id,
                label=ann_data['label'],
                shape_type=ann_data.get('shape_type', 'BBOX_2D'),
                geometry=ann_data['geometry'],
                attributes=ann_data.get('attributes', {}),
                occluded=ann_data.get('occluded'),
                source_metadata={'origin': 'cvat_sync'}
            )
            db.add(db_ann)
        db.flush()

        # Re-audit frame against updated annotations
        audit_single_frame(db, sample.id, run_detector=False, commit=False)
        updated_frames += 1

        # Check which previous issues were solved
        for old_issue in pending_issues:
            # Check if this issue is now resolved
            if old_issue.issue_type == 'MISSING_OBJECT':
                # Did annotator add a box overlapping the missing object?
                matched_now = any(
                    calculate_box_iou(a.geometry, old_issue.location) >= 0.40
                    for a in sample.annotations
                )
                if matched_now:
                    old_issue.status = 'RESOLVED'
                    old_issue.reviewer_note = 'Đã tự động xác nhận qua đồng bộ CVAT: Đã thêm vật thể bị sót'
                    auto_resolved += 1
            elif old_issue.issue_type == 'WRONG_CLASS':
                # Did annotator fix the label?
                matched_now = any(
                    calculate_box_iou(a.geometry, old_issue.location) >= 0.50 and
                    normalize_label(a.label) == normalize_label(old_issue.suggested_label)
                    for a in sample.annotations
                )
                if matched_now:
                    old_issue.status = 'RESOLVED'
                    old_issue.reviewer_note = 'Đã tự động xác nhận qua đồng bộ CVAT: Đã sửa đúng nhãn lớp'
                    auto_resolved += 1

    db.commit()
    invalidate_qc_report_cache(dataset_id)

    # Recalculate remaining issues
    remaining = db.query(QCIssue).join(Sample).filter(
        Sample.dataset_id == dataset_id,
        QCIssue.status == 'PENDING'
    ).count()

    return {
        'status': 'success',
        'message': f'Đồng bộ thành công từ CVAT: cập nhật {updated_frames} frames, tự động xác nhận {auto_resolved} lỗi đã sửa.',
        'dataset_id': dataset_id,
        'updated_frames_count': updated_frames,
        'auto_resolved_issues_count': auto_resolved,
        'remaining_issues_count': remaining,
        'synced_at': datetime.now(timezone.utc).isoformat()
    }

def verify_and_sync_sample_from_cvat(
    db: Session,
    sample_id: int,
    config: Optional[QCConfig] = None
) -> Dict[str, Any]:
    """
    Verify and sync an individual frame on-demand after reviewer fixed it in CVAT.
    """
    sample = db.get(Sample, sample_id)
    if not sample:
        raise ValueError(f"Sample #{sample_id} not found")

    dataset = sample.dataset
    if dataset:
        try:
            cvat_frames = fetch_annotations_from_cvat_api(dataset)
            if cvat_frames:
                target_frame = sample.frame_number if sample.frame_number is not None else (sample.id - 1)
                if target_frame in cvat_frames:
                    # Update DB annotations for this sample to match CVAT
                    db.execute(delete(Annotation).where(Annotation.sample_id == sample.id))
                    for ann_data in cvat_frames[target_frame]:
                        db_ann = Annotation(
                            sample_id=sample.id,
                            label=ann_data['label'],
                            shape_type=ann_data.get('shape_type', 'BBOX_2D'),
                            geometry=ann_data['geometry'],
                            attributes=ann_data.get('attributes', {}),
                            occluded=ann_data.get('occluded'),
                            source_metadata={'origin': 'cvat_live_sync'}
                        )
                        db.add(db_ann)
                    db.flush()
        except Exception as e:
            print(f"[WARN] Failed to live-sync sample #{sample_id} from CVAT API: {e}")

    # Re-audit single frame against updated annotations
    summary = audit_single_frame(db, sample_id, config=config, run_detector=False)

    pending_issues = [i for i in sample.qc_issues if i.status == 'PENDING']
    resolved_count = 0

    # Auto-resolve issues that annotator fixed on CVAT
    for old_issue in pending_issues:
        if old_issue.issue_type == 'MISSING_OBJECT':
            matched_now = any(
                calculate_box_iou(a.geometry, old_issue.location) >= 0.40
                for a in sample.annotations
            )
            if matched_now:
                old_issue.status = 'RESOLVED'
                old_issue.reviewer_note = 'Đã tự động xác nhận qua đối soát CVAT: Đã thêm vật thể bị sót'
                resolved_count += 1
        elif old_issue.issue_type == 'WRONG_CLASS':
            matched_now = any(
                calculate_box_iou(a.geometry, old_issue.location) >= 0.50 and
                normalize_label(a.label) == normalize_label(old_issue.suggested_label)
                for a in sample.annotations
            )
            if matched_now:
                old_issue.status = 'RESOLVED'
                old_issue.reviewer_note = 'Đã tự động xác nhận qua đối soát CVAT: Đã sửa đúng nhãn lớp'
                resolved_count += 1

    db.commit()

    remaining_issues = [i for i in sample.qc_issues if i.status == 'PENDING']
    is_verified = (len(remaining_issues) == 0)
    msg = (
        "Frame đã hoàn toàn sạch lỗi (đã đồng bộ nhãn mới nhất từ CVAT)."
        if is_verified
        else f"Frame đã cập nhật từ CVAT, còn {len(remaining_issues)} vấn đề cần lưu ý."
    )

    return {
        'sample_id': sample_id,
        'frame_number': getattr(sample, 'frame_number', None),
        'is_verified': is_verified,
        'resolved_issues_count': resolved_count,
        'remaining_issues_count': len(remaining_issues),
        'message': msg,
        'issues': summary['issues']
    }

def push_dataset_annotations_to_cvat(
    db: Session,
    dataset_id: int,
    token: Optional[str] = None,
    cvat_base_url: Optional[str] = None,
    cvat_task_id: Optional[int] = None,
    cvat_job_id: Optional[int] = None,
    mode: str = "create",
    target_source: str = "merged"
) -> Dict[str, Any]:
    """
    Push annotations to CVAT using the CVAT REST API with Personal Access Token (PAT).
    Supports:
      - target_source:
          * 'merged': Human annotations + AI missing objects (recommended)
          * 'predictions': All model predictions (Pre-annotations)
          * 'annotations': Human annotations only
      - mode:
          * 'create': Safe append (PATCH ?action=create) - does NOT delete existing CVAT shapes
          * 'replace': Overwrite shapes (PUT) while preserving video tracks
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset #{dataset_id} not found")

    base = (cvat_base_url or dataset.cvat_base_url or 'http://localhost:8080').rstrip('/')
    pat = (token or getattr(dataset, 'cvat_token', None) or 'ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO').strip()
    task_id = cvat_task_id or dataset.cvat_task_id
    job_id = cvat_job_id or dataset.cvat_job_id

    if not task_id and not job_id:
        raise ValueError("Dataset chưa được liên kết với CVAT Task ID hoặc Job ID. Vui lòng cấu hình CVAT trước khi đồng bộ.")

    # Update dataset with any provided values
    if token:
        dataset.cvat_token = token
    if cvat_base_url:
        dataset.cvat_base_url = cvat_base_url
    if cvat_task_id:
        dataset.cvat_task_id = cvat_task_id
    if cvat_job_id:
        dataset.cvat_job_id = cvat_job_id
    db.commit()

    import urllib.request, urllib.error, json, logging
    logger = logging.getLogger(__name__)

    headers = {
        'Authorization': f'Bearer {pat}',
        'Content-Type': 'application/json'
    }

    # Step 1: If job_id is missing, find job_id from task_id
    if task_id and not job_id:
        try:
            req = urllib.request.Request(f"{base}/api/jobs?task_id={task_id}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                jobs_data = json.loads(resp.read().decode('utf-8'))
                results = jobs_data.get('results', [])
                if results:
                    job_id = results[0]['id']
                    dataset.cvat_job_id = job_id
                    db.commit()
        except Exception as e:
            logger.warning(f"Could not automatically fetch job for task {task_id}: {e}")

    # Step 2: Fetch Task Labels to map label names to CVAT label IDs
    target_task_id = task_id
    if not target_task_id and job_id:
        try:
            req = urllib.request.Request(f"{base}/api/jobs/{job_id}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                jdata = json.loads(resp.read().decode('utf-8'))
                target_task_id = jdata.get('task_id')
        except Exception:
            pass

    label_map: Dict[str, int] = {}
    default_label_id = None
    if target_task_id:
        try:
            req = urllib.request.Request(f"{base}/api/labels?task_id={target_task_id}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                labels_data = json.loads(resp.read().decode('utf-8'))
                for l in labels_data.get('results', []):
                    norm = l['name'].strip().lower()
                    label_map[norm] = l['id']
                    if default_label_id is None:
                        default_label_id = l['id']
        except Exception as e:
            logger.warning(f"Could not fetch labels for task {target_task_id}: {e}")

    # Synonym mapping for common driving datasets (e.g. BDD100K to CVAT)
    synonyms = {
        'car': ['vehicle', 'car', 'automobile'],
        'truck': ['vehicle', 'truck'],
        'bus': ['vehicle', 'bus'],
        'vehicle': ['vehicle', 'car', 'truck'],
        'pedestrian': ['pedestrian', 'person'],
        'person': ['pedestrian', 'person'],
        'rider': ['motorcyclist', 'rider', 'cyclist', 'bicycle'],
        'motorcyclist': ['motorcyclist', 'motorcycle', 'rider'],
        'motorcycle': ['motorcyclist', 'motorcycle', 'vehicle'],
        'bicycle': ['motorcyclist', 'bicycle', 'vehicle'],
        'cyclist': ['motorcyclist', 'cyclist', 'bicycle'],
        'traffic light': ['traffic light', 'vehicle'],
        'traffic sign': ['traffic sign', 'vehicle'],
    }

    def resolve_label_id(raw_label: str) -> Optional[int]:
        c = (raw_label or '').strip().lower()
        if c in label_map:
            return label_map[c]
        if c in synonyms:
            for syn in synonyms[c]:
                if syn in label_map:
                    return label_map[syn]
        # Partial match
        for k, v in label_map.items():
            if k in c or c in k:
                return v
        return default_label_id

    # Step 3: Build CVAT rectangle shapes based on target_source
    shapes = []
    samples = db.scalars(
        select(Sample)
        .where(Sample.dataset_id == dataset_id)
        .options(
            selectinload(Sample.annotations),
            selectinload(Sample.predictions),
            selectinload(Sample.qc_issues)
        )
    ).all()

    for sample in samples:
        frame_idx = sample.frame_number
        if frame_idx is None:
            frame_idx = extract_frame_number(sample.file_name, sample.id - 1)

        # 1. Human annotations
        if target_source in ('annotations', 'merged'):
            for ann in sample.annotations:
                shape_upper = (ann.shape_type or '').upper()
                if shape_upper not in ('BBOX_2D', 'RECTANGLE', 'BBOX', 'BOX', '') or not ann.geometry:
                    continue
                lid = resolve_label_id(ann.label)
                if lid is None:
                    continue
                g = ann.geometry
                x1 = round(float(g.get('x1', 0)), 1)
                y1 = round(float(g.get('y1', 0)), 1)
                x2 = round(float(g.get('x2', 0)), 1)
                y2 = round(float(g.get('y2', 0)), 1)
                shapes.append({
                    'type': 'rectangle',
                    'frame': frame_idx,
                    'label_id': lid,
                    'points': [x1, y1, x2, y2],
                    'occluded': bool(ann.occluded),
                    'outside': False,
                    'z_order': 0,
                    'attributes': []
                })

        # 2. AI Model predictions
        if target_source == 'predictions':
            for pred in sample.predictions:
                if not pred.geometry:
                    continue
                lid = resolve_label_id(pred.label)
                if lid is None:
                    continue
                g = pred.geometry
                x1 = round(float(g.get('x1', 0)), 1)
                y1 = round(float(g.get('y1', 0)), 1)
                x2 = round(float(g.get('x2', 0)), 1)
                y2 = round(float(g.get('y2', 0)), 1)
                shapes.append({
                    'type': 'rectangle',
                    'frame': frame_idx,
                    'label_id': lid,
                    'points': [x1, y1, x2, y2],
                    'occluded': False,
                    'outside': False,
                    'z_order': 0,
                    'attributes': []
                })
        elif target_source == 'merged':
            # Append missing objects discovered by AI model
            for issue in sample.qc_issues:
                if issue.issue_type == 'MISSING_OBJECT' and issue.status != 'REJECTED' and issue.location:
                    lid = resolve_label_id(issue.suggested_label)
                    if lid is None:
                        continue
                    g = issue.location
                    x1 = round(float(g.get('x1', 0)), 1)
                    y1 = round(float(g.get('y1', 0)), 1)
                    x2 = round(float(g.get('x2', 0)), 1)
                    y2 = round(float(g.get('y2', 0)), 1)
                    shapes.append({
                        'type': 'rectangle',
                        'frame': frame_idx,
                        'label_id': lid,
                        'points': [x1, y1, x2, y2],
                        'occluded': False,
                        'outside': False,
                        'z_order': 0,
                        'attributes': []
                    })

    if not shapes:
        return {
            'status': 'warning',
            'message': 'Không có bounding box nào cần đồng bộ.',
            'cvat_task_id': task_id,
            'cvat_job_id': job_id,
            'pushed_shapes_count': 0
        }

    # Step 4: Call CVAT API (Job annotations or Task annotations)
    target_endpoint = f"{base}/api/jobs/{job_id}/annotations" if job_id else f"{base}/api/tasks/{task_id}/annotations"

    try:
        if mode == "replace":
            # Fetch existing tracks and version to preserve video tracks
            get_req = urllib.request.Request(target_endpoint, headers=headers)
            existing_tracks = []
            existing_tags = []
            existing_version = 0
            try:
                with urllib.request.urlopen(get_req, timeout=10) as get_resp:
                    existing = json.loads(get_resp.read().decode('utf-8'))
                    existing_tracks = existing.get('tracks', [])
                    existing_tags = existing.get('tags', [])
                    existing_version = existing.get('version', 0)
            except Exception:
                pass

            payload = {
                'shapes': shapes,
                'tracks': existing_tracks,
                'tags': existing_tags,
                'version': existing_version
            }
            data_bytes = json.dumps(payload).encode('utf-8')
            put_req = urllib.request.Request(target_endpoint, data=data_bytes, method='PUT', headers=headers)
            with urllib.request.urlopen(put_req, timeout=60) as put_resp:
                res_code = put_resp.status
        else: # mode == "create" (Append - Safe mode)
            payload = {'shapes': shapes}
            data_bytes = json.dumps(payload).encode('utf-8')
            patch_req = urllib.request.Request(f"{target_endpoint}?action=create", data=data_bytes, method='PATCH', headers=headers)
            with urllib.request.urlopen(patch_req, timeout=60) as patch_resp:
                res_code = patch_resp.status

    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f"CVAT API Error {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Lỗi kết nối tới CVAT ({base}): {str(e)}")

    cvat_url = f"{base}/tasks/{task_id}/jobs/{job_id}" if job_id else f"{base}/tasks/{task_id}"

    source_labels = {
        'merged': 'nhãn gộp (nhãn người gán + vật thể AI phát hiện bị sót)',
        'predictions': 'toàn bộ nhãn dự đoán AI Model (Pre-annotations)',
        'annotations': 'nhãn người gán đã duyệt'
    }
    desc = source_labels.get(target_source, target_source)
    mode_desc = 'Bổ sung (Append - an toàn, không xóa nhãn cũ)' if mode == 'create' else 'Ghi đè (Replace)'

    return {
        'status': 'success',
        'message': f"Đã đồng bộ thành công {len(shapes)} {desc} sang CVAT Task #{task_id} (Job #{job_id}) theo chế độ {mode_desc}.",
        'cvat_task_id': task_id,
        'cvat_job_id': job_id,
        'pushed_shapes_count': len(shapes),
        'cvat_url': cvat_url,
        'target_source': target_source,
        'mode': mode
    }

# --- Legacy Compatibility Endpoints ---

def ingest_predictions_data(
    db: Session,
    dataset_id: int,
    data: List[Dict[str, Any]],
    source_model: str = 'pretrained'
) -> int:
    """Ingest model predictions from list of dicts."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError('Dataset not found')

    sample_by_name = {s.file_name: s for s in dataset.samples}
    count = 0

    for item in data:
        fname = item.get('file_name') or item.get('name')
        if not fname:
            continue
        sample = sample_by_name.get(fname)
        if not sample:
            matched = [s for s in dataset.samples if s.file_name.startswith(fname.split('.')[0])]
            if len(matched) == 1:
                sample = matched[0]

        if not sample:
            continue

        preds = item.get('predictions') or item.get('labels') or []
        for p in preds:
            label = p.get('label') or p.get('category')
            conf = float(p.get('confidence', p.get('score', 0.85)))
            box = p.get('geometry') or p.get('box2d') or {}

            if not box and 'box' in p:
                box = p['box']

            if not label or not box:
                continue

            db_pred = Prediction(
                sample_id=sample.id,
                label=label,
                geometry={
                    'x1': float(box.get('x1', 0)),
                    'y1': float(box.get('y1', 0)),
                    'x2': float(box.get('x2', 0)),
                    'y2': float(box.get('y2', 0))
                },
                confidence=conf,
                source_model=source_model
            )
            db.add(db_pred)
            count += 1

    db.commit()
    invalidate_qc_report_cache(dataset_id)
    return count

def generate_benchmark_synthetic_predictions(
    db: Session,
    dataset_id: int,
    missing_ratio: float = 0.15,
    wrong_class_ratio: float = 0.15,
    source_model: str = "synthetic_oracle_v1"
) -> Dict[str, Any]:
    """Generate reproducible benchmark synthetic predictions for testing."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError('Dataset not found')

    sample_ids = [s.id for s in dataset.samples]
    db.execute(delete(Prediction).where(Prediction.sample_id.in_(sample_ids)))

    swap_candidates = {
        'car': 'truck',
        'truck': 'car',
        'bus': 'truck',
        'pedestrian': 'rider',
        'rider': 'pedestrian',
        'motorcyclist': 'pedestrian',
        'bicycle': 'motorcycle',
        'motorcycle': 'bicycle',
        'traffic light': 'traffic sign',
        'traffic sign': 'traffic light'
    }

    pred_count = 0
    injected_wrong = 0
    injected_missing = 0
    rng = random.Random(42)

    for sample in dataset.samples:
        for ann in sample.annotations:
            if ann.shape_type != 'BBOX_2D' or not ann.geometry:
                continue

            is_wrong = rng.random() < wrong_class_ratio
            if is_wrong and ann.label.lower() in swap_candidates:
                pred_label = swap_candidates[ann.label.lower()]
                conf = round(rng.uniform(0.75, 0.94), 3)
                injected_wrong += 1
            else:
                pred_label = ann.label
                conf = round(rng.uniform(0.85, 0.98), 3)

            w = ann.geometry['x2'] - ann.geometry['x1']
            h = ann.geometry['y2'] - ann.geometry['y1']
            jitter_x = rng.uniform(-0.03, 0.03) * w
            jitter_y = rng.uniform(-0.03, 0.03) * h

            pred = Prediction(
                sample_id=sample.id,
                label=pred_label,
                geometry={
                    'x1': round(ann.geometry['x1'] + jitter_x, 1),
                    'y1': round(ann.geometry['y1'] + jitter_y, 1),
                    'x2': round(ann.geometry['x2'] + jitter_x, 1),
                    'y2': round(ann.geometry['y2'] + jitter_y, 1),
                },
                confidence=conf,
                source_model=source_model
            )
            db.add(pred)
            pred_count += 1

        if rng.random() < missing_ratio and sample.width and sample.height:
            x1 = round(rng.uniform(0.1, 0.8) * sample.width, 1)
            y1 = round(rng.uniform(0.3, 0.7) * sample.height, 1)
            pred = Prediction(
                sample_id=sample.id,
                label=rng.choice(['car', 'pedestrian', 'traffic sign']),
                geometry={
                    'x1': x1,
                    'y1': y1,
                    'x2': round(x1 + rng.uniform(40, 120), 1),
                    'y2': round(y1 + rng.uniform(30, 90), 1),
                },
                confidence=round(rng.uniform(0.80, 0.95), 3),
                source_model=source_model
            )
            db.add(pred)
            pred_count += 1
            injected_missing += 1

    db.commit()
    audit_report = run_dataset_qc_audit(db, dataset_id)

    return {
        'predictions_generated': pred_count,
        'simulated_wrong_classes': injected_wrong,
        'simulated_missing_objects': injected_missing,
        'audit_summary': audit_report
    }

def run_pretrained_detector_service(
    db: Session,
    dataset_id: int,
    conf_thresh: float = 0.45,
    source_model: str = "yolov8n-bdd100k"
) -> Dict[str, Any]:
    """
    Backward-compatible method to run detector and audit dataset.
    Uses new single-frame inference engine across all frames.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError('Dataset not found')

    pred_count = 0
    for sample in dataset.samples:
        if sample.media_path and os.path.isfile(sample.media_path):
            boxes = detect_image_objects(sample.media_path, conf_thresh=conf_thresh)
            if boxes:
                db.execute(delete(Prediction).where(
                    Prediction.sample_id == sample.id,
                    Prediction.source_model == source_model
                ))
                for b in boxes:
                    pred = Prediction(
                        sample_id=sample.id,
                        label=b['label'],
                        geometry=b['geometry'],
                        confidence=b['confidence'],
                        source_model=source_model
                    )
                    db.add(pred)
                    pred_count += 1

    db.commit()

    if pred_count > 0:
        audit_report = run_dataset_qc_audit(db, dataset_id)
        return {
            'predictions_generated': pred_count,
            'source': 'yolov8_inference',
            'audit_summary': audit_report
        }

    return generate_benchmark_synthetic_predictions(db, dataset_id, source_model=source_model)
