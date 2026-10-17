import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from app.models import Dataset, Sample, Annotation, Prediction
from app.schemas.experiments import WBFExperimentRequest
from app.services.wbf_experiments import ARMS, perturb_image, consolidate_views, run_experiment


BOX = dict(x1=4., y1=4., x2=28., y2=28.)


def fake_runner(path, conf_thresh, models, timings=None):
    assert Path(path).is_file()
    if timings is not None:
        timings.update({m: .01 for m in models})
    return {m: [dict(label='car', confidence=.95, geometry=BOX)] for m in models}


def fixture_frame(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (40, 40), (90, 110, 140)).save(path)
    return dict(file_name=path.name, image_path=str(path),
                annotations=[dict(id=1, shape_type='BBOX_2D', label='truck', geometry=BOX)],
                golden=[dict(label='car', geometry=BOX)])


def test_perturbations_reproducible_and_keep_coordinate_space():
    image = Image.fromarray(np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3))
    config = WBFExperimentRequest()
    kinds = []
    for view in range(4):
        a, meta = perturb_image(image, view, config)
        b, _ = perturb_image(image, view, config)
        assert np.array_equal(a, b)
        assert a.size == image.size
        kinds.append(meta['kind'])
        if view:
            assert not np.array_equal(a, image)
    assert kinds == ['original', 'gaussian', 'brightness', 'blur']
    noise, _ = perturb_image(image, 1, config)
    other, _ = perturb_image(image, 1, config.model_copy(update={'seed': 43}))
    assert not np.array_equal(noise, other)


def test_repeated_views_remain_one_physical_model_vote():
    from app.services.ensemble import weighted_box_fusion
    repeated = [[dict(label='car', confidence=.95, geometry=BOX)]] * 4
    consolidated = consolidate_views(repeated, 'yolo26n', .55, .35)
    fused = weighted_box_fusion({'yolo26n': consolidated})
    assert len(fused) == 1
    assert fused[0].num_models_agreed == 1
    assert fused[0].tier == 3  # Four passes must never masquerade as four independent detectors.


def test_identical_clean_passes_preserve_distinct_overlapping_objects():
    boxes = [dict(label='car', confidence=.95, geometry=BOX),
             dict(label='car', confidence=.9, geometry={**BOX, 'x1': 6., 'x2': 30.})]
    consolidated = consolidate_views([boxes, boxes, boxes, boxes], 'yolo26n', .55, .35)
    assert len(consolidated) == 2
    assert [b.geometry for b in consolidated] == [b['geometry'] for b in boxes]


def test_six_arms_count_actual_calls_and_use_independent_golden(tmp_path):
    frame = fixture_frame(tmp_path)
    output = tmp_path / 'report.json'
    config = WBFExperimentRequest(repeats=4)
    calls = []
    def runner(*args, **kwargs):
        calls.append(tuple(kwargs['models']))
        return fake_runner(*args, **kwargs)
    report = run_experiment([frame], config, output, runner=runner)
    assert report['status'] == 'COMPLETED', report.get('error')
    assert list(report['summary']) == list(ARMS)
    assert [report['summary'][arm]['detector_calls'] for arm in ARMS] == [1, 4, 4, 3, 12, 12]
    assert sum(map(len, calls)) == 3 + 36  # Explicit warmup plus all six arms, no cached-prediction reuse.
    for row in report['summary'].values():
        assert row['detection_metrics']['f1'] == 1
        assert row['qc_metrics']['WRONG_CLASS']['recall'] == 1
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'COMPLETED'
    assert frame['annotations'][0]['label'] == 'truck'  # Evaluation never applies the suggested fix.


def test_no_golden_or_annotation_does_not_invent_quality_or_qc_errors(tmp_path):
    frame = fixture_frame(tmp_path)
    frame.update(golden=None, annotations=[], annotations_available=False)
    report = run_experiment([frame], WBFExperimentRequest(), tmp_path / 'report.json', runner=fake_runner)
    assert report['status'] == 'COMPLETED'
    for row in report['summary'].values():
        assert row['detection_metrics'] is None and row['qc_metrics'] is None
        assert row['qc_issues'] is None
        assert row['agreement_with_single_once']['reference_is_ground_truth'] is False


def test_real_inference_failure_is_recorded_without_synthetic_fallback(tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('Weights unavailable')
    report = run_experiment([fixture_frame(tmp_path)], WBFExperimentRequest(), tmp_path / 'failed.json', runner=fail)
    assert report['status'] == 'ERROR'
    assert report['error'] == 'Weights unavailable'
    assert not report['frames'] and not report['summary']


@pytest.mark.parametrize('update', [dict(repeats=1), dict(sample_limit=31), dict(gaussian_sigma=float('nan')),
                                    dict(models=['yolo26n', 'yolo26n']), dict(models=['yolo26n', 'yolo26s'])])
def test_experiment_rejects_invalid_or_fake_independent_sources(update):
    with pytest.raises(ValidationError):
        WBFExperimentRequest(**update)


def test_api_run_report_and_storage_isolation(client, db_session, tmp_path, monkeypatch):
    from app.routers import experiments as routes
    class InlineExecutor:
        def submit(self, work):
            work()
    monkeypatch.setattr(routes, 'REPORT_ROOT', tmp_path / 'reports')
    monkeypatch.setattr(routes, '_executor', InlineExecutor())
    monkeypatch.setattr(routes, 'run_experiment', lambda frames, config, path, progress:
                        run_experiment(frames, config, path, progress=progress, runner=fake_runner))
    frame = fixture_frame(tmp_path)
    dataset = Dataset(name='Experiment isolation', format='CVAT', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name=frame['file_name'], media_path=frame['image_path'],
                    task_type='BBOX_2D', width=40, height=40)
    db_session.add(sample); db_session.flush()
    db_session.add(Annotation(sample_id=sample.id, label='truck', shape_type='BBOX_2D', geometry=BOX))
    db_session.add(Prediction(sample_id=sample.id, label='bus', source_model='existing', confidence=.8, geometry=BOX))
    db_session.commit()
    prefix = f'/api/datasets/{dataset.id}/ensemble/experiments'
    response = client.post(prefix, json={'sample_limit': 1})
    assert response.status_code == 202, response.text
    run_id = response.json()['id']
    status = client.get(f'{prefix}/{run_id}').json()
    assert status['status'] == 'COMPLETED' and 'frames' not in status
    assert len(client.get(prefix).json()) == 1
    report = client.get(f'{prefix}/{run_id}/report')
    assert report.status_code == 200 and len(report.json()['frames']) == 1
    assert client.get(f'{prefix}/invalid-id').status_code == 404
    assert client.post(prefix, json={'benchmark_id': 999}).status_code == 422
    db_session.expire_all()
    saved = db_session.get(Sample, sample.id)
    assert saved.annotations[0].label == 'truck'
    assert len(saved.predictions) == 1 and saved.predictions[0].source_model == 'existing'
    assert not saved.qc_issues


def test_api_rejects_missing_media_and_concurrent_jobs(client, db_session, tmp_path, monkeypatch):
    from app.routers import experiments as routes
    dataset = Dataset(name='Missing media', format='CVAT', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='missing.jpg', task_type='BBOX_2D')
    db_session.add(sample); db_session.commit()
    prefix = f'/api/datasets/{dataset.id}/ensemble/experiments'
    assert client.post(prefix, json={}).status_code == 422
    image = tmp_path / 'image.png'
    Image.new('RGB', (40, 40)).save(image)
    sample.media_path = str(image); db_session.commit()
    monkeypatch.setattr(routes, '_active', {'existing': {'status': 'RUNNING'}})
    assert client.post(prefix, json={}).status_code == 409
