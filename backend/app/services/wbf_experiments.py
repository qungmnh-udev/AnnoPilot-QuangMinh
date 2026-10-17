"""Real inference experiments. Isolated reports; never changes operational labels/predictions."""
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.schemas.qc import QCConfig
from app.services.ensemble import DetectionBox, weighted_box_fusion
from app.services.evaluation import true_errors, match_issue_pairs, metric
from app.services.qc_engine import detect_sample_qc_issues, normalize_label
from app.services.candidates import iou
from app.services.triad_detector import detect_triad_frame


ARMS = ('single_model_once', 'single_model_repeated_clean', 'single_model_repeated_perturbed',
        'multi_model_once', 'multi_model_repeated_clean', 'multi_model_repeated_perturbed')


def perturb_image(image, view, config, frame_index=0):
    """Photometric changes only: boxes stay in original pixel coordinates."""
    if view == 0:
        return image.copy(), {'kind': 'original'}
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, frame_index, view]))
    kind = ('gaussian', 'brightness', 'blur')[(view - 1) % 3]
    if kind == 'gaussian':
        data = np.asarray(image, dtype=np.float32)
        data = np.clip(data + rng.normal(0, config.gaussian_sigma, data.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(data), {'kind': kind, 'sigma_8bit': config.gaussian_sigma}
    if kind == 'brightness':
        factor = 1 + config.brightness_delta * (1 if (view // 3) % 2 == 0 else -1)
        return ImageEnhance.Brightness(image).enhance(factor), {'kind': kind, 'factor': factor}
    return image.filter(ImageFilter.GaussianBlur(config.blur_radius)), {'kind': kind, 'radius': config.blur_radius}


def detection_dict(box):
    return dict(label=box.label, confidence=box.confidence, geometry=box.geometry)


def consolidate_views(views, physical_model, threshold, confidence):
    """Fuse views within one detector, then expose ONE physical-model vote."""
    # Identical clean passes carry no new evidence. Preserve distinct overlapping detections
    # instead of letting the extra temporal clustering merge them as a side effect.
    if all(boxes == views[0] for boxes in views):
        return [DetectionBox(source_model=physical_model, **{k: b[k] for k in ('label', 'confidence')}, **b['geometry'])
                for b in views[0]]
    per_view = {f'view_{index}': [DetectionBox(source_model=f'view_{index}',
                label=b['label'], confidence=b['confidence'], **b['geometry']) for b in boxes]
                for index, boxes in enumerate(views)}
    return [DetectionBox(source_model=physical_model, label=b.label, confidence=b.confidence, **b.geometry)
            for b in weighted_box_fusion(per_view, iou_threshold=threshold, confidence_threshold=confidence)]


def detection_counts(predictions, golden, threshold=.5):
    edges = sorted([(iou(p['geometry'], g['geometry']), pi, gi)
                    for pi, p in enumerate(predictions) for gi, g in enumerate(golden)
                    if normalize_label(p['label']) == normalize_label(g['label'])], reverse=True)
    used_p, used_g, overlaps = set(), set(), []
    for overlap, pi, gi in edges:
        if overlap >= threshold and pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); overlaps.append(overlap)
    return [len(used_p), len(predictions) - len(used_p), len(golden) - len(used_g)], overlaps


def evaluate_frame(frame, detections, model_boxes, config):
    annotation_available = frame.get('annotations_available', True)
    annotations = [SimpleNamespace(**a) for a in frame.get('annotations', [])]
    predictions = [SimpleNamespace(id=None, source_model=model, extra_data={}, **detection_dict(b))
                   for model, boxes in model_boxes.items() for b in boxes]
    qc_config = QCConfig(strategy='baseline' if len(model_boxes) == 1 else 'multi_model',
                         baseline_model=config.baseline_model, expected_models=list(model_boxes))
    issues = detect_sample_qc_issues(0, annotations, predictions, qc_config) if predictions and annotation_available else []
    result = {'qc_issue_count': len(issues) if annotation_available else None,
              'qc_issues': issues, 'detection_counts': None, 'qc_counts': None}
    if frame.get('golden') is not None:
        result['detection_counts'], result['matched_ious'] = detection_counts(detections, frame['golden'])
        if not annotation_available:
            return result
        errors = true_errors(annotations, frame['golden'], .5, qc_config.taxonomy_mapping)
        totals = {}
        for kind in ('MISSING_OBJECT', 'WRONG_CLASS'):
            warned = [i for i in issues if i['issue_type'] == kind]
            actual = [e for e in errors if e['issue_type'] == kind]
            tp = len(match_issue_pairs(warned, actual, .5))
            totals[kind] = [tp, len(warned) - tp, len(actual) - tp]
        result['qc_counts'] = totals
    return result


def summarize(frames, has_golden):
    summary = {}
    for arm in ARMS:
        values = [frame['arms'][arm] for frame in frames]
        row = dict(frame_count=len(values), detections=sum(len(v['predictions']) for v in values),
                   detector_calls=sum(v['detector_calls'] for v in values),
                   inference_seconds=sum(v['inference_seconds'] for v in values),
                   fusion_seconds=sum(v['fusion_seconds'] for v in values),
                   qc_seconds=sum(v['qc_seconds'] for v in values),
                   qc_issues=sum(v['evaluation']['qc_issue_count'] for v in values)
                       if all(v['evaluation']['qc_issue_count'] is not None for v in values) else None,
                   detection_metrics=None, qc_metrics=None)
        if has_golden:
            counts = [sum(v['evaluation']['detection_counts'][i] for v in values) for i in range(3)]
            row['detection_metrics'] = metric(*counts)
            overlaps = [x for v in values for x in v['evaluation'].get('matched_ious', [])]
            row['detection_metrics']['mean_matched_iou'] = sum(overlaps) / len(overlaps) if overlaps else None
            if all(v['evaluation']['qc_counts'] is not None for v in values):
                row['qc_metrics'] = {kind: metric(*[sum(v['evaluation']['qc_counts'][kind][i] for v in values)
                                                   for i in range(3)]) for kind in ('MISSING_OBJECT', 'WRONG_CLASS')}
        stability_counts = [0, 0, 0]
        for frame in frames:
            counts, _ = detection_counts(frame['arms'][arm]['predictions'], frame['arms']['single_model_once']['predictions'])
            stability_counts = [a + b for a, b in zip(stability_counts, counts)]
        row['agreement_with_single_once'] = metric(*stability_counts)
        row['agreement_with_single_once']['reference_is_ground_truth'] = False
        summary[arm] = row
    reference_seconds = summary['single_model_once']['inference_seconds']
    for row in summary.values():
        row['inference_seconds_per_frame'] = row['inference_seconds'] / row['frame_count'] if row['frame_count'] else None
        row['relative_inference_cost'] = row['inference_seconds'] / reference_seconds if reference_seconds else None
    return summary


def write_report(path, report):
    """Atomic checkpoint so polling never reads partially written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def run_experiment(frames, config, output_path, progress=None, runner=None):
    runner = runner or detect_triad_frame
    real_inference = runner is detect_triad_frame
    if real_inference:
        from app.services.model_registry import checkpoint_scope
        def scoped_runner(*args, **kwargs):
            with checkpoint_scope(frames[0].get('dataset_id') if frames else None):
                return detect_triad_frame(*args, **kwargs)
        runner = scoped_runner
    report = dict(status='RUNNING', created_at=datetime.now(timezone.utc).isoformat(), config=config.model_dump(), frames=[], summary={},
                  total_frames=len(frames), processed_frames=0, warmup_seconds=0,
                  error=None, quality_measured=bool(frames) and all(f.get('golden') is not None for f in frames),
                  notes=['Repeated views are not independent detectors. Each physical model has one final vote.',
                         'Agreement with baseline is consistency, not accuracy. Quality metrics require independent Golden.',
                         'CPU wall time includes preprocessing inside detectors; warm-up is separate. No operational dataset is modified.'])
    write_report(output_path, report)
    started = perf_counter()
    try:
        if not frames:
            raise ValueError('No images selected')
        if real_inference:
            from app.services.provenance import detector_provenance
            from app.services.yolo_detector import get_yolo_weights_path, inference_device
            from app.services.triad_detector import get_rtdetr_weights_path
            import torch
            checkpoints = {config.baseline_model: get_yolo_weights_path(config.baseline_model),
                           'rtdetr_l': get_rtdetr_weights_path(),
                           'fasterrcnn_mobilenet_v3': Path(torch.hub.get_dir()) / 'checkpoints' / 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'}
            from app.services.model_registry import active_checkpoint
            for model in config.models:
                trained = active_checkpoint(model, frames[0].get('dataset_id'))
                if trained:
                    checkpoints[model] = Path(trained['path'])
            report['provenance'] = {m: detector_provenance(checkpoints[m]) for m in config.models}
            report['runtime'] = {'yolo_device': inference_device(), 'torch_threads': torch.get_num_threads(),
                                 'platform': platform.platform()}
        warm_started = perf_counter()
        runner(frames[0]['image_path'], conf_thresh=config.confidence, models=config.models)
        report['warmup_seconds'] = perf_counter() - warm_started
        with tempfile.TemporaryDirectory(prefix='wbf-views-', dir=Path(output_path).parent) as temp:
            for index, frame in enumerate(frames):
                with Image.open(frame['image_path']) as source:
                    image = source.convert('RGB')
                result = dict(file_name=frame['file_name'], image_sha256=hashlib.sha256(Path(frame['image_path']).read_bytes()).hexdigest(),
                              width=image.width, height=image.height, annotations=frame.get('annotations', []),
                              annotations_available=frame.get('annotations_available', True), golden=frame.get('golden'), arms={})
                for arm in ARMS:
                    models = config.models if arm.startswith('multi_') else [config.baseline_model]
                    repeats = config.repeats if 'repeated' in arm else 1
                    views = {m: [] for m in models}
                    raw_views, infer_time = [], 0
                    for view in range(repeats):
                        perturbed = arm.endswith('_perturbed')
                        current, transform = perturb_image(image, view if perturbed else 0, config, index)
                        # Lossless PNG preserves the generated noise. Same pixels for each clean control pass.
                        image_path = Path(temp) / f'{index}-{arm}-{view}.png'
                        current.save(image_path)
                        timings = {}
                        per_model = runner(str(image_path), conf_thresh=config.confidence, models=models, timings=timings)
                        infer_time += sum(timings.values())
                        for m in models:
                            views[m].append(per_model[m])
                        raw_views.append(dict(view=view, transform=transform, timings=timings, predictions=per_model))
                    fusion_started = perf_counter()
                    per_model = {m: consolidate_views(views[m], m, config.fusion_iou, config.confidence) for m in models}
                    if arm.startswith('single_'):
                        detections = [detection_dict(b) for b in per_model[config.baseline_model]]
                    else:
                        fused = weighted_box_fusion(per_model, model_weights={m: 1.0 for m in models},
                                                    iou_threshold=config.fusion_iou, confidence_threshold=config.confidence)
                        detections = [dict(**detection_dict(b), tier=b.tier, evidence=b.evidence) for b in fused]
                    fusion_time = perf_counter() - fusion_started
                    qc_started = perf_counter()
                    evaluation = evaluate_frame(frame, detections, per_model, config)
                    result['arms'][arm] = dict(predictions=detections, views=raw_views, detector_calls=repeats * len(models),
                                              inference_seconds=infer_time, fusion_seconds=fusion_time,
                                              qc_seconds=perf_counter() - qc_started, evaluation=evaluation)
                    if progress:
                        progress(dict(processed_frames=index, total_frames=len(frames), current_frame=frame['file_name'], arm=arm))
                report['frames'].append(result)
                report['processed_frames'] = index + 1
                report['summary'] = summarize(report['frames'], report['quality_measured'])
                report['elapsed_seconds'] = perf_counter() - started
                write_report(output_path, report)
        report['status'] = 'COMPLETED'
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = str(exc)
    report['elapsed_seconds'] = perf_counter() - started
    write_report(output_path, report)
    return report
