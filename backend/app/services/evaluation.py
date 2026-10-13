"""Independent golden annotations define true errors; consensus never defines GT."""
from time import perf_counter
import math
from app.models import GoldenBenchmark, QCEvaluation, AcceptanceEvent
from app.services.candidates import iou
from app.services.qc_engine import detect_sample_qc_issues, normalize_label


def annotations_snapshot(dataset):
    return {s.file_name: [dict(id=a.id, label=a.label, shape_type=a.shape_type,
                              geometry=a.geometry) for a in s.annotations] for s in dataset.samples}


def create_benchmark(db, dataset, request):
    names = {s.file_name: s for s in dataset.samples}
    submitted = [s.file_name for s in request.samples]
    if len(set(submitted)) != len(submitted) or any(n not in names for n in submitted):
        raise ValueError('Golden frames must be unique and belong to this dataset')
    if not request.verified_by.strip():
        raise ValueError('Independent verifier is required')
    for frame in request.samples:
        sample = names[frame.file_name]
        for a in frame.annotations:
            if (sample.width and a.geometry['x2'] > sample.width) or (sample.height and a.geometry['y2'] > sample.height):
                raise ValueError('Golden box exceeds original image dimensions')
    benchmark = GoldenBenchmark(dataset_id=dataset.id, verified_by=request.verified_by.strip(),
                                notes=request.notes, samples=[s.model_dump() for s in request.samples])
    db.add(benchmark)
    db.commit()
    return dict(id=benchmark.id, frame_count=len(benchmark.samples), verified_by=benchmark.verified_by)


def true_errors(annotations, golden, threshold, mapping):
    edges = sorted([(iou(a.geometry, g['geometry']), ai, gi) for ai, a in enumerate(annotations)
                    if a.shape_type == 'BBOX_2D' for gi, g in enumerate(golden)], reverse=True)
    used_a, used_g, errors = set(), set(), []
    for overlap, ai, gi in edges:
        if overlap < threshold or ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        g = golden[gi]
        if normalize_label(annotations[ai].label, mapping) != normalize_label(g['label'], mapping):
            errors.append(dict(issue_type='WRONG_CLASS', location=g['geometry'], label=normalize_label(g['label'], mapping)))
    for gi, g in enumerate(golden):
        if gi not in used_g:
            errors.append(dict(issue_type='MISSING_OBJECT', location=g['geometry'], label=normalize_label(g['label'], mapping)))
    return errors


def match_issue_pairs(issues, errors, threshold):
    edges = sorted([(iou(issue['location'], error['location']), ii, ei)
                    for ii, issue in enumerate(issues) for ei, error in enumerate(errors)
                    if issue['issue_type'] == error['issue_type']], reverse=True)
    matched_i, matched_e = set(), set()
    pairs = []
    for overlap, ii, ei in edges:
        if overlap >= threshold and ii not in matched_i and ei not in matched_e:
            matched_i.add(ii)
            matched_e.add(ei)
            pairs.append((ii, ei))
    return pairs


def count_matches(issues, errors, threshold, mapping):
    pairs = match_issue_pairs(issues, errors, threshold)
    return len(pairs), len(issues) - len(pairs), len(errors) - len(pairs)


def metric(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def compare_strategies(db, dataset, request):
    benchmark = db.get(GoldenBenchmark, request.benchmark_id)
    if not benchmark or benchmark.dataset_id != dataset.id:
        raise ValueError('Benchmark does not belong to this dataset')
    sample_map = {s.file_name: s for s in dataset.samples}
    snapshots = []
    for frame in benchmark.samples:
        sample = sample_map.get(frame['file_name'])
        if sample is None:
            raise ValueError('Benchmark frame is no longer available')
        preds = [p for p in sample.predictions if p.source_model in request.pilot_models]
        identities = [(sample.inference_state or {}).get(m, {}).get('checkpoint_sha256') for m in request.pilot_models]
        known = [identity for identity in identities if identity]
        if len(set(known)) != len(known):
            raise ValueError('Pilot sources share an identical checkpoint; do not count aliases as independent detectors')
        for model in request.pilot_models:
            state = (sample.inference_state or {}).get(model, {})
            if state.get('status') == 'ERROR':
                raise ValueError(f"Inference failed for {model} on {sample.file_name}")
            if not any(p.source_model == model for p in preds) and state.get('status') not in ('SUCCESS', 'IMPORTED'):
                raise ValueError(f"Missing inference/import coverage for {model} on {sample.file_name}; empty results must be explicitly recorded")
            if model.startswith(('ensemble_', 'synthetic')) or any((p.extra_data or {}).get('synthetic') for p in preds):
                raise ValueError('Evaluation requires real independent detector predictions')
        snapshots.append(dict(file_name=sample.file_name, annotations=[dict(id=a.id, label=a.label, geometry=a.geometry, shape_type=a.shape_type) for a in sample.annotations],
                              predictions=[dict(id=p.id, label=p.label, geometry=p.geometry, confidence=p.confidence,
                                                source_model=p.source_model, provenance=p.extra_data or {}) for p in preds],
                              inference_state=sample.inference_state or {}, golden=frame['annotations']))
    results = []
    for strategy in ('baseline', 'multi_model'):
        config = request.config.model_copy(update={'strategy': strategy, 'baseline_model': request.baseline_model,
                                                   'expected_models': request.pilot_models if strategy == 'multi_model' else [request.baseline_model]})
        totals = {kind: [0, 0, 0] for kind in ('MISSING_OBJECT', 'WRONG_CLASS')}
        correct_suggestions = {kind: 0 for kind in totals}
        started = perf_counter()
        frame_results = []
        for frame, snap in zip(benchmark.samples, snapshots):
            sample = sample_map[frame['file_name']]
            predictions = [p for p in sample.predictions if p.source_model in request.pilot_models]
            if strategy == 'baseline':
                predictions = [p for p in predictions if p.source_model == request.baseline_model]
            # Empty successful baseline inference is a valid measurement with FN.
            issues = detect_sample_qc_issues(sample.id, sample.annotations, predictions, config) if predictions else []
            errors = true_errors(sample.annotations, frame['annotations'], request.evaluation_iou, config.taxonomy_mapping)
            for kind in totals:
                predicted = [i for i in issues if i['issue_type'] == kind]
                actual = [e for e in errors if e['issue_type'] == kind]
                pairs = match_issue_pairs(predicted, actual, request.evaluation_iou)
                counts = (len(pairs), len(predicted) - len(pairs), len(actual) - len(pairs))
                totals[kind] = [a + b for a, b in zip(totals[kind], counts)]
                correct_suggestions[kind] += sum(normalize_label(predicted[ii]['suggested_label'], config.taxonomy_mapping) == actual[ei]['label'] for ii, ei in pairs)
            frame_results.append(dict(file_name=sample.file_name, issues=issues, true_errors=errors))
        metrics = {kind: metric(*counts) for kind, counts in totals.items()}
        for kind in totals:
            metrics[kind]['correct_class_suggestions'] = correct_suggestions[kind]
            metrics[kind]['class_suggestion_accuracy'] = correct_suggestions[kind] / totals[kind][0] if totals[kind][0] else None
        overall = [sum(totals[k][index] for k in totals) for index in range(3)]
        metrics['overall'] = metric(*overall)
        metrics['frame_count'] = len(snapshots)
        metrics['qc_seconds'] = perf_counter() - started  # excludes inference and human review
        selected_models = [request.baseline_model] if strategy == 'baseline' else request.pilot_models
        durations = [snap['inference_state'].get(m, {}).get('inference_seconds') for snap in snapshots for m in selected_models]
        metrics['inference_seconds'] = sum(durations) if all(isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 for d in durations) else None
        minutes = request.review_minutes.get(strategy)
        metrics['review_minutes'] = minutes
        metrics['true_errors_per_review_hour'] = overall[0] * 60 / minutes if minutes else None
        run = QCEvaluation(dataset_id=dataset.id, benchmark_id=benchmark.id, strategy=strategy,
                           config={**config.model_dump(), 'evaluation_iou': request.evaluation_iou, 'pilot_models': request.pilot_models},
                           input_snapshot={'frames': snapshots, 'results': frame_results}, metrics=metrics)
        db.add(run)
        db.flush()
        results.append(dict(id=run.id, benchmark_id=run.benchmark_id, strategy=strategy, metrics=metrics,
                            created_at=run.created_at.isoformat()))
    db.commit()
    return results


def record_acceptance(db, dataset, request):
    previous = sorted(dataset.acceptance_events, key=lambda e: e.id)
    last = previous[-1] if previous else None
    if request.status == 'SUBMITTED':
        if last and last.status != 'RETURNED':
            raise ValueError('Submit only a new dataset or a returned submission')
        if last and last.strategy != request.strategy:
            raise ValueError('Keep the same strategy across an acceptance cycle; use a separate dataset for the other strategy')
        cycle = (last.cycle + 1) if last else 1
    else:
        if not last or last.status != 'SUBMITTED' or last.strategy != request.strategy:
            raise ValueError('Return/accept requires a submitted version with the same strategy')
        cycle = last.cycle
    event = AcceptanceEvent(dataset_id=dataset.id, cycle=cycle, strategy=request.strategy,
                            status=request.status, actor=request.actor.strip(), reason=request.reason,
                            annotation_snapshot=annotations_snapshot(dataset))
    db.add(event)
    db.commit()
    db.expire(dataset, ['acceptance_events'])
    return acceptance_summary(dataset)


def acceptance_summary(dataset):
    events = sorted(dataset.acceptance_events, key=lambda e: e.id)
    return dict(status=events[-1].status if events else 'NOT_SUBMITTED',
                strategy=events[-1].strategy if events else None,
                submission_count=sum(e.status == 'SUBMITTED' for e in events),
                return_count=sum(e.status == 'RETURNED' for e in events),
                events=[dict(id=e.id, cycle=e.cycle, status=e.status, strategy=e.strategy,
                             actor=e.actor, reason=e.reason, created_at=e.created_at.isoformat()) for e in events])
