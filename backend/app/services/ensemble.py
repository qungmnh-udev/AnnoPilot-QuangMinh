"""
Multi-Model Ensemble Engine for AnnoPilot Pre-labeling.

Implements Weighted Box Fusion (WBF) to combine predictions from multiple
object detection models into high-quality consensus pre-labels.

Architecture:
  Model A (e.g. YOLOv8x)   ──┐
                               ├─→ WBF Fusion → Tri-Tier Classification
  Model B (e.g. RT-DETR)    ──┘

Tri-Tier output:
  Tier 1 (High Consensus)     → Auto-accept pre-label
  Tier 2 (Class Disagreement) → Flag for 1-click human review
  Tier 3 (Model Divergence)   → Route to QC queue for manual inspection
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import delete

from app.models import Sample, Prediction, Dataset
from app.services.qc_engine import (
    calculate_box_iou,
    calculate_box_area,
    normalize_label,
    invalidate_qc_report_cache,
    run_dataset_qc_audit,
)


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class DetectionBox:
    """A single detection from one model."""
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    source_model: str = "unknown"

    @property
    def geometry(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class FusedBox:
    """Result of fusing one or more overlapping detections."""
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    tier: int  # 1 = high consensus, 2 = class disagreement, 3 = divergence
    tier_reason: str = ""
    contributing_models: List[str] = field(default_factory=list)
    model_votes: Dict[str, float] = field(default_factory=dict)  # label → avg conf
    num_models_agreed: int = 0

    @property
    def geometry(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


# ────────────────────────────────────────────────────────────────────
# Weighted Box Fusion (WBF) – Pure Python implementation
# ────────────────────────────────────────────────────────────────────

def _iou(a: DetectionBox, b: DetectionBox) -> float:
    """IoU between two DetectionBox instances."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    model_predictions: Dict[str, List[DetectionBox]],
    model_weights: Optional[Dict[str, float]] = None,
    iou_threshold: float = 0.55,
    confidence_threshold: float = 0.30,
    high_consensus_conf: float = 0.85,
    high_consensus_iou: float = 0.60,
) -> List[FusedBox]:
    """
    Weighted Box Fusion across multiple model outputs.

    For each cluster of overlapping boxes (IoU ≥ iou_threshold):
      - Fuse coordinates via weighted average (weight = model_weight × confidence)
      - Classify into Tier 1/2/3 based on consensus

    Args:
        model_predictions: {model_name: [DetectionBox, ...]}
        model_weights: {model_name: weight} — defaults to equal weights
        iou_threshold: IoU threshold for clustering
        confidence_threshold: minimum fused confidence to keep
        high_consensus_conf: confidence bar for Tier 1
        high_consensus_iou: IoU bar for Tier 1

    Returns:
        List of FusedBox sorted by confidence descending.
    """
    model_names = list(model_predictions.keys())
    num_models = len(model_names)

    if num_models == 0:
        return []

    # Default equal weights
    if model_weights is None:
        model_weights = {m: 1.0 for m in model_names}

    # Flatten all boxes with their model weight
    all_boxes: List[Tuple[DetectionBox, float]] = []
    for model_name, boxes in model_predictions.items():
        w = model_weights.get(model_name, 1.0)
        for box in boxes:
            all_boxes.append((box, w))

    # Sort by weighted confidence descending
    all_boxes.sort(key=lambda x: x[0].confidence * x[1], reverse=True)

    # Greedy clustering
    clusters: List[List[Tuple[DetectionBox, float]]] = []
    used = [False] * len(all_boxes)

    for i, (box_i, w_i) in enumerate(all_boxes):
        if used[i]:
            continue
        cluster = [(box_i, w_i)]
        used[i] = True

        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            box_j, w_j = all_boxes[j]
            # Check IoU against the cluster representative (first / highest-conf box)
            if _iou(box_i, box_j) >= iou_threshold:
                cluster.append((box_j, w_j))
                used[j] = True

        clusters.append(cluster)

    # Fuse each cluster
    fused_results: List[FusedBox] = []

    for cluster in clusters:
        # Weighted average coordinates
        total_weight = 0.0
        wx1, wy1, wx2, wy2 = 0.0, 0.0, 0.0, 0.0
        contributing_models = set()
        label_confs: Dict[str, List[float]] = {}  # label → [conf, ...]
        model_labels: Dict[str, str] = {}  # model → predicted label

        for box, w in cluster:
            effective_w = w * box.confidence
            total_weight += effective_w
            wx1 += box.x1 * effective_w
            wy1 += box.y1 * effective_w
            wx2 += box.x2 * effective_w
            wy2 += box.y2 * effective_w
            contributing_models.add(box.source_model)
            norm_label = normalize_label(box.label)
            label_confs.setdefault(norm_label, []).append(box.confidence)
            model_labels[box.source_model] = norm_label

        if total_weight <= 0:
            continue

        fused_x1 = wx1 / total_weight
        fused_y1 = wy1 / total_weight
        fused_x2 = wx2 / total_weight
        fused_y2 = wy2 / total_weight

        # Determine winning label and confidence
        model_votes: Dict[str, float] = {}
        for lbl, confs in label_confs.items():
            model_votes[lbl] = round(sum(confs) / len(confs), 3)

        best_label = max(model_votes, key=model_votes.get)  # type: ignore
        # Fused confidence: average of all contributing confidences
        all_confs = [box.confidence for box, _ in cluster]
        fused_conf = sum(all_confs) / len(all_confs)

        if fused_conf < confidence_threshold:
            continue

        # Determine tier
        n_models = len(contributing_models)
        all_same_label = len(label_confs) == 1
        min_conf_in_cluster = min(all_confs)

        # Compute average IoU within cluster pairs
        avg_iou = 0.0
        pair_count = 0
        boxes_in_cluster = [b for b, _ in cluster]
        for ci in range(len(boxes_in_cluster)):
            for cj in range(ci + 1, len(boxes_in_cluster)):
                avg_iou += _iou(boxes_in_cluster[ci], boxes_in_cluster[cj])
                pair_count += 1
        if pair_count > 0:
            avg_iou /= pair_count

        if n_models >= 2 and all_same_label and min_conf_in_cluster >= high_consensus_conf and avg_iou >= high_consensus_iou:
            tier = 1
            tier_reason = f"High consensus: {n_models} models agree on '{best_label}' (avg IoU={avg_iou:.2f}, min conf={min_conf_in_cluster:.2f})"
        elif n_models >= 2 and not all_same_label:
            tier = 2
            labels_str = ", ".join(f"{lbl}: {conf:.0%}" for lbl, conf in sorted(model_votes.items(), key=lambda x: -x[1]))
            tier_reason = f"Class disagreement: [{labels_str}]"
        elif n_models >= 2 and all_same_label:
            # Both models agree on label but IoU or confidence is below Tier 1 threshold
            tier = 1
            tier_reason = f"Moderate consensus: {n_models} models agree on '{best_label}' (avg IoU={avg_iou:.2f})"
        else:
            # Only 1 model saw this object
            tier = 3
            src = list(contributing_models)[0]
            tier_reason = f"Only detected by {src} (conf={fused_conf:.2f})"

        fused_results.append(FusedBox(
            label=best_label,
            confidence=round(fused_conf, 3),
            x1=fused_x1,
            y1=fused_y1,
            x2=fused_x2,
            y2=fused_y2,
            tier=tier,
            tier_reason=tier_reason,
            contributing_models=sorted(contributing_models),
            model_votes=model_votes,
            num_models_agreed=n_models,
        ))

    fused_results.sort(key=lambda f: (-f.tier, -f.confidence))
    # Sort: Tier 1 first, then Tier 2, then Tier 3, highest confidence first within tier
    fused_results.sort(key=lambda f: (f.tier, -f.confidence))

    return fused_results


# ────────────────────────────────────────────────────────────────────
# Multi-Model Ensemble Service for AnnoPilot
# ────────────────────────────────────────────────────────────────────

def load_predictions_from_json(
    data: List[Dict[str, Any]],
    source_model: str = "external",
) -> Dict[str, List[DetectionBox]]:
    """
    Parse a JSON predictions list into {file_name: [DetectionBox, ...]}.
    Supports both AnnoPilot format and BDD100K format.
    """
    result: Dict[str, List[DetectionBox]] = {}

    for item in data:
        fname = item.get("file_name") or item.get("name") or ""
        if not fname:
            continue

        preds = item.get("predictions") or item.get("labels") or []
        boxes: List[DetectionBox] = []

        for p in preds:
            label = p.get("label") or p.get("category") or ""
            conf = float(p.get("confidence", p.get("score", 0.5)))
            box = p.get("geometry") or p.get("box2d") or p.get("box") or {}

            if not label or not box:
                continue

            boxes.append(DetectionBox(
                label=label,
                confidence=conf,
                x1=float(box.get("x1", 0)),
                y1=float(box.get("y1", 0)),
                x2=float(box.get("x2", 0)),
                y2=float(box.get("y2", 0)),
                source_model=source_model,
            ))

        if boxes:
            result[fname] = boxes

    return result


def fuse_multi_model_predictions(
    models_data: Dict[str, Dict[str, List[DetectionBox]]],
    model_weights: Optional[Dict[str, float]] = None,
    iou_threshold: float = 0.55,
    confidence_threshold: float = 0.30,
    high_consensus_conf: float = 0.85,
    high_consensus_iou: float = 0.60,
) -> Dict[str, List[FusedBox]]:
    """
    Fuse predictions from multiple models, grouped by file name.

    Args:
        models_data: {model_name: {file_name: [DetectionBox, ...]}}
        model_weights: {model_name: weight}
        iou_threshold: WBF IoU threshold
        confidence_threshold: minimum fused confidence
        high_consensus_conf: confidence threshold for Tier 1
        high_consensus_iou: IoU threshold for Tier 1

    Returns:
        {file_name: [FusedBox, ...]}
    """
    # Collect all file names across all models
    all_files = set()
    for model_preds in models_data.values():
        all_files.update(model_preds.keys())

    fused_by_file: Dict[str, List[FusedBox]] = {}

    for fname in sorted(all_files):
        # Gather per-model predictions for this file
        per_model: Dict[str, List[DetectionBox]] = {}
        for model_name, preds_by_file in models_data.items():
            if fname in preds_by_file:
                per_model[model_name] = preds_by_file[fname]

        fused = weighted_box_fusion(
            per_model,
            model_weights=model_weights,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            high_consensus_conf=high_consensus_conf,
            high_consensus_iou=high_consensus_iou,
        )
        if fused:
            fused_by_file[fname] = fused

    return fused_by_file


def run_ensemble_and_store(
    db: Session,
    dataset_id: int,
    models_data: Dict[str, Dict[str, List[DetectionBox]]],
    model_weights: Optional[Dict[str, float]] = None,
    iou_threshold: float = 0.55,
    confidence_threshold: float = 0.30,
    high_consensus_conf: float = 0.85,
    high_consensus_iou: float = 0.60,
    ensemble_source_model: str = "ensemble_wbf_v1",
    run_audit: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end: fuse multi-model predictions, store in DB as Predictions,
    and optionally re-run QC audit.

    Returns summary dict with counts and tier breakdown.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    # Build file_name → sample lookup
    sample_by_name: Dict[str, Sample] = {}
    for s in dataset.samples:
        sample_by_name[s.file_name] = s
        # Also index by basename without extension for fuzzy match
        base = s.file_name.rsplit(".", 1)[0] if "." in s.file_name else s.file_name
        sample_by_name.setdefault(base, s)

    # Fuse
    fused_by_file = fuse_multi_model_predictions(
        models_data,
        model_weights=model_weights,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        high_consensus_conf=high_consensus_conf,
        high_consensus_iou=high_consensus_iou,
    )

    # Clear previous ensemble predictions
    sample_ids = [s.id for s in dataset.samples]
    if sample_ids:
        db.execute(delete(Prediction).where(
            Prediction.sample_id.in_(sample_ids),
            Prediction.source_model == ensemble_source_model
        ))

    # Store fused boxes
    total_stored = 0
    tier_counts = {1: 0, 2: 0, 3: 0}

    for fname, fused_boxes in fused_by_file.items():
        sample = sample_by_name.get(fname)
        if not sample:
            # Try fuzzy match
            base = fname.rsplit(".", 1)[0] if "." in fname else fname
            sample = sample_by_name.get(base)
        if not sample:
            continue

        for fb in fused_boxes:
            pred = Prediction(
                sample_id=sample.id,
                label=fb.label,
                geometry=fb.geometry,
                confidence=fb.confidence,
                source_model=ensemble_source_model,
                extra_data={
                    "tier": fb.tier,
                    "tier_reason": fb.tier_reason,
                    "contributing_models": fb.contributing_models,
                    "model_votes": fb.model_votes,
                    "num_models_agreed": fb.num_models_agreed,
                }
            )
            db.add(pred)
            total_stored += 1
            tier_counts[fb.tier] = tier_counts.get(fb.tier, 0) + 1

    db.commit()
    invalidate_qc_report_cache(dataset_id)

    result: Dict[str, Any] = {
        "status": "success",
        "ensemble_source_model": ensemble_source_model,
        "models_used": list(models_data.keys()),
        "total_fused_predictions": total_stored,
        "tier_breakdown": {
            "tier_1_high_consensus": tier_counts.get(1, 0),
            "tier_2_class_disagreement": tier_counts.get(2, 0),
            "tier_3_model_divergence": tier_counts.get(3, 0),
        },
        "files_processed": len(fused_by_file),
    }

    if run_audit:
        audit_report = run_dataset_qc_audit(db, dataset_id)
        result["audit_summary"] = audit_report

    return result


def run_ensemble_from_existing_predictions(
    db: Session,
    dataset_id: int,
    model_names: Optional[List[str]] = None,
    model_weights: Optional[Dict[str, float]] = None,
    iou_threshold: float = 0.55,
    confidence_threshold: float = 0.30,
    high_consensus_conf: float = 0.85,
    high_consensus_iou: float = 0.60,
    ensemble_source_model: str = "ensemble_wbf_v1",
    run_audit: bool = True,
) -> Dict[str, Any]:
    """
    Run WBF ensemble using predictions already stored in the database
    from different source_model values.

    If model_names is None, automatically discovers all distinct source_model
    values in the dataset's predictions (excluding previous ensemble results).
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    # Discover available models
    all_preds: List[Prediction] = []
    for sample in dataset.samples:
        all_preds.extend(sample.predictions)

    if model_names is None:
        model_names = sorted(set(
            p.source_model for p in all_preds
            if p.source_model and not p.source_model.startswith("ensemble_")
        ))

    if len(model_names) < 1:
        return {
            "status": "error",
            "message": "Cần ít nhất 1 nguồn dự đoán (source_model) để chạy ensemble. "
                       "Hãy tải lên predictions từ nhiều model trước.",
            "models_found": [],
        }

    # Group predictions by model → file_name → [DetectionBox]
    models_data: Dict[str, Dict[str, List[DetectionBox]]] = {m: {} for m in model_names}

    for sample in dataset.samples:
        for pred in sample.predictions:
            if pred.source_model not in models_data:
                continue
            geo = pred.geometry or {}
            box = DetectionBox(
                label=pred.label,
                confidence=pred.confidence,
                x1=float(geo.get("x1", 0)),
                y1=float(geo.get("y1", 0)),
                x2=float(geo.get("x2", 0)),
                y2=float(geo.get("y2", 0)),
                source_model=pred.source_model,
            )
            models_data[pred.source_model].setdefault(sample.file_name, []).append(box)

    return run_ensemble_and_store(
        db=db,
        dataset_id=dataset_id,
        models_data=models_data,
        model_weights=model_weights,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        high_consensus_conf=high_consensus_conf,
        high_consensus_iou=high_consensus_iou,
        ensemble_source_model=ensemble_source_model,
        run_audit=run_audit,
    )


def get_dataset_prediction_models(db: Session, dataset_id: int) -> Dict[str, Any]:
    """
    Retrieve all distinct model sources and prediction counts for a dataset.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    sample_ids = [s.id for s in dataset.samples]
    if not sample_ids:
        return {
            "dataset_id": dataset_id,
            "models": [],
            "counts": {},
            "has_ensemble": False,
        }

    from sqlalchemy import func
    rows = (
        db.query(Prediction.source_model, func.count(Prediction.id))
        .filter(Prediction.sample_id.in_(sample_ids))
        .group_by(Prediction.source_model)
        .all()
    )

    counts = {r[0] or "unknown": int(r[1]) for r in rows}
    models = sorted(counts.keys())
    has_ensemble = any(m.startswith("ensemble_") for m in models)

    return {
        "dataset_id": dataset_id,
        "models": models,
        "counts": counts,
        "has_ensemble": has_ensemble,
    }

