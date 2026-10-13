"""Spatial correspondence shared by pre-label generation and QC.

Matching is class-agnostic. Every original detection is retained, but a model
gets at most one vote per candidate. Scores are heuristics, not probabilities.
"""
from dataclasses import dataclass, field
import math


def iou(a, b):
    inter = max(0, min(a['x2'], b['x2']) - max(a['x1'], b['x1'])) * max(0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
    area = lambda g: max(0, g['x2'] - g['x1']) * max(0, g['y2'] - g['y1'])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


@dataclass
class Candidate:
    detections: list
    votes: list
    label: str
    confidence: float
    geometry: dict
    tier: int
    tier_reason: str
    evidence: dict = field(default_factory=dict)


def build_candidates(detections, model_names, weights=None, iou_threshold=.55,
                     high_consensus_conf=.85, high_consensus_iou=.60):
    weights = weights or {}
    for model in model_names:
        w = weights.get(model, 1.0)
        if not math.isfinite(w) or w <= 0:
            raise ValueError('Model weights must be finite and positive')
    valid = []
    for d in detections:
        g = d['geometry']
        values = [g.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
        if (any(v is None or not math.isfinite(v) for v in values)
                or g['x2'] <= g['x1'] or g['y2'] <= g['y1']
                or not math.isfinite(d['confidence']) or not 0 <= d['confidence'] <= 1
                or not d['label']):
            raise ValueError('Invalid detection geometry, label or confidence')
        valid.append(d)
    remaining = sorted(valid, key=lambda d: (-d['confidence'] * weights.get(d['model_id'], 1), d['model_id'], d['label']))
    result = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        rest = []
        for d in remaining:
            (cluster if iou(seed['geometry'], d['geometry']) >= iou_threshold else rest).append(d)
        remaining = rest
        by_model = {}
        for d in cluster:
            by_model.setdefault(d['model_id'], d)  # highest-confidence detection
        votes = list(by_model.values())
        support = {}
        counts = {}
        for d in votes:
            support[d['label']] = support.get(d['label'], 0) + weights.get(d['model_id'], 1) * d['confidence']
            counts[d['label']] = counts.get(d['label'], 0) + 1
        label = sorted(support, key=lambda c: (-support[c], -counts[c], c))[0]
        selected = [d for d in votes if d['label'] == label]
        total = sum(weights.get(d['model_id'], 1) * d['confidence'] for d in selected)
        if total <= 0:
            geometry = dict(seed['geometry'])
        else:
            geometry = {k: round(sum(d['geometry'][k] * weights.get(d['model_id'], 1) * d['confidence'] for d in selected) / total, 2) for k in seed['geometry']}
        confidence = sum(d['confidence'] * weights.get(d['model_id'], 1) for d in selected) / sum(weights.get(d['model_id'], 1) for d in selected)
        pairs = [iou(a['geometry'], b['geometry']) for index, a in enumerate(votes) for b in votes[index + 1:]]
        spatial = sum(pairs) / len(pairs) if pairs else 0.0
        if len(support) > 1:
            tier, reason = 2, 'Class disagreement; human verification required'
        elif len(votes) >= 2 and min(d['confidence'] for d in votes) >= high_consensus_conf and spatial >= high_consensus_iou:
            tier, reason = 1, 'High consensus; not independent ground truth'
        elif len(votes) == 1:
            tier, reason = 3, f"Only detected by {votes[0]['model_id']}"
        else:
            tier, reason = 3, 'Moderate consensus below high-consensus thresholds'
        contributing = sorted(by_model)
        evidence = {
            'per_model_predictions': cluster,
            'contributing_models': contributing,
            'absent_models': sorted(set(model_names) - set(contributing)),
            'detection_count': len(votes),
            'num_models_agreed': counts[label],
            'model_votes': {c: round(s / sum(weights.get(d['model_id'], 1) for d in votes), 4) for c, s in support.items()},
            'class_vote_counts': counts,
            'spatial_consistency': round(spatial, 4),
            'consensus_tier': tier,
            'tier_reason': reason,
            'score_calibrated': False,
        }
        result.append(Candidate(cluster, votes, label, round(confidence, 4), geometry, tier, reason, evidence))
    return result
