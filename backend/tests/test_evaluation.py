import pytest
from app.models import Dataset, Sample, Annotation, Prediction, QCEvaluation

BOX = dict(x1=10., y1=10., x2=90., y2=90.)
FAR = dict(x1=150., y1=100., x2=200., y2=200.)


def setup_benchmark(client, db_session):
    dataset = Dataset(name='Golden comparison', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='gold.jpg', task_type='BBOX_2D', width=300, height=300)
    db_session.add(sample); db_session.flush()
    db_session.add(Annotation(sample_id=sample.id, label='bus', shape_type='BBOX_2D', geometry=BOX))
    db_session.add_all([
        Prediction(sample_id=sample.id, source_model='baseline', label='bus', geometry=BOX, confidence=.9),
        Prediction(sample_id=sample.id, source_model='independent', label='truck', geometry=BOX, confidence=.95),
        Prediction(sample_id=sample.id, source_model='independent', label='pedestrian', geometry=FAR, confidence=.9),
    ])
    db_session.commit()
    response = client.post(f'/api/datasets/{dataset.id}/qc/benchmarks', json={
        'verified_by': 'Independent reviewer', 'includes_unwarned_regions': True,
        'samples': [{'file_name': 'gold.jpg', 'annotations': [{'label': 'truck', 'geometry': BOX}, {'label': 'pedestrian', 'geometry': FAR}]}],
    })
    assert response.status_code == 200
    return dataset.id, sample.id, response.json()['id']


def test_independent_benchmark_measures_blindspots_and_persists_snapshots(client, db_session):
    dataset_id, sample_id, benchmark_id = setup_benchmark(client, db_session)
    result = client.post(f'/api/datasets/{dataset_id}/qc/evaluations/compare', json={
        'benchmark_id': benchmark_id, 'baseline_model': 'baseline', 'pilot_models': ['baseline', 'independent'],
        'review_minutes': {'multi_model': 10},
    })
    assert result.status_code == 200, result.text
    baseline, pilot = result.json()
    assert baseline['metrics']['MISSING_OBJECT']['recall'] == 0
    assert baseline['metrics']['WRONG_CLASS']['recall'] == 0
    assert baseline['metrics']['overall']['fn'] == 2
    assert baseline['metrics']['overall']['precision'] is None
    assert pilot['metrics']['overall']['tp'] == 2
    assert pilot['metrics']['overall']['fp'] == 0
    assert pilot['metrics']['overall']['recall'] == 1
    assert pilot['metrics']['true_errors_per_review_hour'] == 12
    assert baseline['metrics']['true_errors_per_review_hour'] is None
    first = client.get(f"/api/datasets/{dataset_id}/qc/evaluations/{baseline['id']}").json()
    second = client.get(f"/api/datasets/{dataset_id}/qc/evaluations/{pilot['id']}").json()
    assert first['input_snapshot']['frames'] == second['input_snapshot']['frames']
    db_session.expire_all()
    assert db_session.get(Sample, sample_id).annotations[0].label == 'bus'
    assert not db_session.get(Sample, sample_id).qc_issues  # evaluation never changes operational review


def test_zero_detection_frame_is_evaluated_and_shared_omission_counts_as_fn(client, db_session):
    dataset_id, sample_id, benchmark_id = setup_benchmark(client, db_session)
    sample = db_session.get(Sample, sample_id)
    for p in list(sample.predictions):
        db_session.delete(p)
    sample.inference_state = {m: {'status': 'SUCCESS'} for m in ('baseline', 'independent')}
    db_session.commit()
    result = client.post(f'/api/datasets/{dataset_id}/qc/evaluations/compare', json={
        'benchmark_id': benchmark_id, 'baseline_model': 'baseline', 'pilot_models': ['baseline', 'independent'],
    })
    assert result.status_code == 200
    assert all(r['metrics']['overall']['fn'] == 2 for r in result.json())


def test_missing_model_coverage_is_rejected(client, db_session):
    dataset_id, _, benchmark_id = setup_benchmark(client, db_session)
    result = client.post(f'/api/datasets/{dataset_id}/qc/evaluations/compare', json={
        'benchmark_id': benchmark_id, 'baseline_model': 'baseline', 'pilot_models': ['baseline', 'unknown'],
    })
    assert result.status_code == 422


def test_acceptance_tracks_actual_return_cycles_and_validates_transitions(client, db_session):
    dataset_id, _, _ = setup_benchmark(client, db_session)
    endpoint = f'/api/datasets/{dataset_id}/qc/acceptance'
    body = {'strategy': 'multi_model', 'actor': 'Reviewer'}
    assert client.post(endpoint, json={**body, 'status': 'RETURNED', 'reason': 'Missing pedestrian'}).status_code == 409
    assert client.post(endpoint, json={**body, 'status': 'SUBMITTED'}).status_code == 200
    assert client.post(endpoint, json={**body, 'status': 'SUBMITTED'}).status_code == 409
    assert client.post(endpoint, json={**body, 'status': 'RETURNED'}).status_code == 422
    assert client.post(endpoint, json={**body, 'status': 'RETURNED', 'reason': 'Missing pedestrian'}).status_code == 200
    assert client.post(endpoint, json={**body, 'status': 'SUBMITTED'}).status_code == 200
    response = client.post(endpoint, json={**body, 'status': 'ACCEPTED'})
    assert response.status_code == 200
    assert response.json()['return_count'] == 1
    assert response.json()['submission_count'] == 2
    assert len(response.json()['events']) == 4


def test_benchmark_requires_independent_full_region_attestation(client, db_session):
    dataset_id, _, _ = setup_benchmark(client, db_session)
    response = client.post(f'/api/datasets/{dataset_id}/qc/benchmarks', json={
        'verified_by': 'Reviewer', 'samples': [{'file_name': 'gold.jpg', 'annotations': []}],
    })
    assert response.status_code == 422


def test_repeated_same_issue_predictions_only_count_once():
    from app.services.evaluation import count_matches
    issues = [dict(issue_type='MISSING_OBJECT', location=BOX, suggested_label='car')] * 2
    errors = [dict(issue_type='MISSING_OBJECT', location=BOX, label='car')]
    assert count_matches(issues, errors, .5, {}) == (1, 1, 0)


def test_detecting_wrong_class_is_separate_from_suggesting_correct_class():
    from app.services.evaluation import count_matches
    issues = [dict(issue_type='WRONG_CLASS', location=BOX, suggested_label='car')]
    errors = [dict(issue_type='WRONG_CLASS', location=BOX, label='truck')]
    assert count_matches(issues, errors, .5, {}) == (1, 0, 0)


def test_aliases_of_identical_checkpoint_are_not_independent_models(client, db_session):
    dataset_id, sample_id, benchmark_id = setup_benchmark(client, db_session)
    sample = db_session.get(Sample, sample_id)
    sample.inference_state = {m: {'status': 'SUCCESS', 'checkpoint_sha256': 'same-checkpoint'} for m in ('baseline', 'independent')}
    db_session.commit()
    response = client.post(f'/api/datasets/{dataset_id}/qc/evaluations/compare', json={
        'benchmark_id': benchmark_id, 'baseline_model': 'baseline', 'pilot_models': ['baseline', 'independent'],
    })
    assert response.status_code == 422
    assert 'identical checkpoint' in response.json()['detail']


def test_synthetic_sandbox_does_not_overwrite_real_predictions(client, db_session):
    dataset_id, sample_id, _ = setup_benchmark(client, db_session)
    before = client.get(f'/api/samples/{sample_id}').json()['predictions']
    response = client.post(f'/api/datasets/{dataset_id}/qc/synthetic-benchmark')
    assert response.status_code == 422
    after = client.get(f'/api/samples/{sample_id}').json()['predictions']
    assert before == after


@pytest.mark.parametrize('payload', [{'weights': {'yolov8n': 0}}, {'models': []}, {'models': ['yolov8n', 'yolov8n']}, {'models': ['unsupported']}])
def test_invalid_pilot_configuration_is_rejected_before_inference(client, payload):
    response = client.post('/api/datasets/999999/ensemble/run-triad', json=payload)
    assert response.status_code == 422
