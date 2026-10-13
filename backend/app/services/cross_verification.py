"""QC reads original detector evidence, independently of pre-label selection."""
from app.services.candidates import build_candidates, iou


def cross_verify(sample_id, annotations, predictions, config):
    from app.services.qc_engine import normalize_label, calculate_box_area, get_confusion_weight
    raw = [p for p in predictions if not (p.source_model or '').startswith('ensemble_')]
    fused = [p for p in predictions if (p.source_model or '').startswith('ensemble_')]
    records = []
    for p in raw:
        if not p.geometry:
            continue
        records.append({'model_id': p.source_model or 'unknown', 'prediction_id': p.id,
                        'label': normalize_label(p.label, config.taxonomy_mapping),
                        'confidence': p.confidence, 'geometry': p.geometry,
                        'provenance': p.extra_data or {}})
    # Uploaded fused predictions may retain snapshots without original DB rows.
    if not records:
        for p in fused:
            records.extend((p.extra_data or {}).get('per_model_predictions', []))
    if not records:
        return None  # legacy single/fused prediction fallback
    models = sorted(set(config.expected_models or []) | {d['model_id'] for d in records})
    if len(models) < 2:
        return None
    candidates = build_candidates(records, models, config.model_weights, config.candidate_iou_threshold)
    gt = [a for a in annotations if a.shape_type == 'BBOX_2D' and a.geometry]
    used = set()
    issues = []
    for c in sorted(candidates, key=lambda c: -max(d['confidence'] for d in c.votes)):
        available = [(iou(a.geometry, c.geometry), a) for a in gt if a.id not in used]
        overlap, annotation = max(available, key=lambda x: x[0]) if available else (0, None)
        evidence = dict(c.evidence)
        evidence['strategy'] = 'multi_model'
        evidence['selected_for_prelabel'] = any(iou(p.geometry, c.geometry) >= config.candidate_iou_threshold for p in fused)
        evidence['prelabel_decision_known'] = bool(fused)
        evidence['requires_human_review'] = True
        evidence['source_model'] = 'cross_verification'
        if annotation is not None and overlap >= config.iou_match_threshold:
            used.add(annotation.id)
            human = normalize_label(annotation.label, config.taxonomy_mapping)
            alternatives = [d for d in c.votes if d['label'] != human and d['confidence'] >= config.min_conf_wrong_class]
            if not alternatives:
                continue
            # Suggest a differing class even when fusion selected the current class.
            alternative = max(alternatives, key=lambda d: (sum(v['confidence'] * (config.model_weights or {}).get(v['model_id'], 1) for v in alternatives if v['label'] == d['label']), d['confidence']))
            score = overlap * alternative['confidence'] * get_confusion_weight(human, alternative['label'])
            evidence.update(iou=round(overlap, 4), model_confidence=alternative['confidence'],
                            reason=f"Models disagree with annotation '{annotation.label}'; inspect alternative '{alternative['label']}'. Consensus is not ground truth.")
            kind, location, ann_id, human_label = 'WRONG_CLASS', annotation.geometry, annotation.id, annotation.label
        else:
            nearby = max((iou(a.geometry, d['geometry']) for a in gt for d in c.votes), default=0)
            reliable = [d for d in c.votes if d['confidence'] >= config.min_conf_missing and calculate_box_area(d['geometry']) >= config.min_box_area]
            if nearby > config.iou_unmatched_threshold or not reliable:
                continue
            alternative = max(reliable, key=lambda d: d['confidence'])
            area = calculate_box_area(alternative['geometry'])
            score = alternative['confidence'] * (1 - nearby) * config.class_reliability.get(alternative['label'], .85)
            evidence.update(max_iou_with_any_annotation=round(nearby, 4), model_confidence=alternative['confidence'],
                            box_area=area, reason='Original detector candidate has no nearby annotation; retained for review even if fusion did not select it.')
            kind, location, ann_id, human_label = 'MISSING_OBJECT', alternative['geometry'], None, None
        prediction_id = alternative.get('prediction_id')
        # Snapshot IDs may belong to another database; only link current rows.
        if prediction_id not in {p.id for p in raw}:
            prediction_id = None
        issues.append(dict(sample_id=sample_id, issue_type=kind, location=location,
                           human_label=human_label, suggested_label=alternative['label'],
                           annotation_id=ann_id, prediction_id=prediction_id,
                           qc_score=round(min(1, max(0, score)), 4), evidence=evidence,
                           status='PENDING', reviewer_note=''))
    return sorted(issues, key=lambda issue: -issue['qc_score'])
