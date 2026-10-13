import pytest
from app.models import Dataset, Sample, Annotation, Prediction
from app.schemas.qc import QCConfig
from app.services.ensemble import DetectionBox, weighted_box_fusion
from app.services.qc_engine import detect_sample_qc_issues, audit_single_frame, get_dataset_qc_report

BOX = dict(x1=10., y1=10., x2=90., y2=90.)


def pred(index, model, label, confidence=.9, geometry=None):
    return Prediction(id=index, sample_id=1, source_model=model, label=label,
                      confidence=confidence, geometry=geometry or BOX, extra_data={})


def test_disagreement_survives_annotation_matching_fused_class():
    annotations = [Annotation(id=1, sample_id=1, label='bus', shape_type='BBOX_2D', geometry=BOX)]
    predictions = [pred(1, 'a', 'bus'), pred(2, 'b', 'bus'), pred(3, 'c', 'truck', .95),
                   pred(4, 'ensemble_wbf_v1', 'bus')]
    issues = detect_sample_qc_issues(1, annotations, predictions)
    assert len(issues) == 1
    assert issues[0]['suggested_label'] == 'truck'
    assert issues[0]['evidence']['consensus_tier'] == 2
    assert len(issues[0]['evidence']['per_model_predictions']) == 3
    assert detect_sample_qc_issues(1, annotations, predictions, QCConfig(strategy='baseline', baseline_model='a')) == []


def test_candidate_not_selected_by_fusion_still_generates_missing_issue():
    far = dict(x1=200., y1=200., x2=250., y2=300.)
    annotations = [Annotation(id=1, label='car', shape_type='BBOX_2D', geometry=BOX)]
    predictions = [pred(1, 'a', 'car'), pred(2, 'b', 'car'), pred(3, 'a', 'pedestrian', .7, far),
                   pred(4, 'ensemble_wbf_v1', 'car')]
    issues = detect_sample_qc_issues(1, annotations, predictions)
    assert len(issues) == 1 and issues[0]['issue_type'] == 'MISSING_OBJECT'
    assert not issues[0]['evidence']['selected_for_prelabel']
    assert issues[0]['evidence']['absent_models'] == ['b']


def test_one_model_gets_one_vote_and_moderate_consensus_is_not_tier_one():
    boxes = {'a': [DetectionBox('truck', .9, **BOX, source_model='a')] * 5,
             'b': [DetectionBox('bus', .8, **BOX, source_model='b')],
             'c': [DetectionBox('bus', .8, **BOX, source_model='c')]}
    fused = weighted_box_fusion(boxes)
    assert len(fused) == 1 and fused[0].label == 'bus'
    assert fused[0].num_models_agreed == 2
    assert fused[0].evidence['detection_count'] == 3
    assert len(fused[0].evidence['per_model_predictions']) == 7
    moderate = weighted_box_fusion({m: [DetectionBox('car', .4, **BOX, source_model=m)] for m in ('a', 'b')})
    assert moderate[0].tier == 3


def test_weights_change_class_vote_and_invalid_weights_are_rejected():
    boxes = {m: [DetectionBox(label, .9, **BOX, source_model=m)] for m, label in [('a', 'bus'), ('b', 'truck')]}
    assert weighted_box_fusion(boxes, {'a': 1, 'b': 3})[0].label == 'truck'
    with pytest.raises(ValueError):
        weighted_box_fusion(boxes, {'a': float('nan')})


def test_empty_inference_replaces_old_predictions(db_session, monkeypatch):
    dataset = Dataset(name='Empty inference', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='empty.jpg', task_type='BBOX_2D', media_path='unused.jpg')
    db_session.add(sample); db_session.flush()
    db_session.add(Prediction(sample_id=sample.id, label='car', geometry=BOX, confidence=.9, source_model='baseline'))
    db_session.commit()
    monkeypatch.setattr('app.services.qc_engine.os.path.isfile', lambda _: True)
    monkeypatch.setattr('app.services.qc_engine.detect_image_objects', lambda *args, **kwargs: [])
    audit_single_frame(db_session, sample.id, source_model='baseline')
    assert not sample.predictions
    assert sample.inference_state['baseline']['status'] == 'SUCCESS'


def test_inference_failure_is_not_a_clean_frame(db_session, monkeypatch):
    dataset = Dataset(name='Failed inference', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='error.jpg', task_type='BBOX_2D', media_path='unused.jpg')
    db_session.add(sample); db_session.commit()
    monkeypatch.setattr('app.services.qc_engine.os.path.isfile', lambda _: True)
    def fail(*args, **kwargs):
        raise RuntimeError('checkpoint unavailable')
    monkeypatch.setattr('app.services.qc_engine.detect_image_objects', fail)
    with pytest.raises(RuntimeError):
        audit_single_frame(db_session, sample.id)
    report = get_dataset_qc_report(db_session, dataset.id, force_refresh=True)
    assert report['inference_error_samples_count'] == 1
    assert report['clean_samples_count'] == 0


def test_review_survives_replacing_prediction_ids(client, db_session):
    dataset = Dataset(name='Review persistence', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='review.jpg', task_type='BBOX_2D')
    db_session.add(sample); db_session.commit()
    payload = {'source_model': 'a', 'predictions': [{'file_name': 'review.jpg', 'predictions': [{'label': 'car', 'confidence': .9, 'geometry': BOX}]}]}
    assert client.post(f'/api/datasets/{dataset.id}/qc/predictions', json=payload).status_code == 200
    issue = client.get(f'/api/samples/{sample.id}').json()['qc_issues'][0]
    endpoint = f"/api/qc/issues/{issue['id']}"
    assert client.put(endpoint, json={'status': 'REJECTED', 'reviewer_note': 'Reflection, not a car'}).status_code == 200
    assert client.post(f'/api/datasets/{dataset.id}/qc/predictions', json=payload).status_code == 200
    issues = client.get(f'/api/samples/{sample.id}').json()['qc_issues']
    assert len(issues) == 1 and issues[0]['status'] == 'REJECTED'
    assert len(client.get(f'/api/samples/{sample.id}/qc/review-history').json()) == 1


def test_accept_retry_does_not_duplicate_annotation(client, db_session):
    dataset = Dataset(name='Accept retry', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='accept.jpg', task_type='BBOX_2D')
    db_session.add(sample); db_session.commit()
    payload = {'source_model': 'a', 'predictions': [{'file_name': 'accept.jpg', 'predictions': [{'label': 'car', 'confidence': .9, 'geometry': BOX}]}]}
    client.post(f'/api/datasets/{dataset.id}/qc/predictions', json=payload)
    issue = client.get(f'/api/samples/{sample.id}').json()['qc_issues'][0]
    for _ in range(2):
        assert client.put(f"/api/qc/issues/{issue['id']}", json={'status': 'ACCEPTED', 'reviewer_note': 'Verified'}).status_code == 200
    assert len(client.get(f'/api/samples/{sample.id}').json()['annotations']) == 1


def test_triad_results_are_committed_per_frame(db_session, monkeypatch):
    from app.services.triad_detector import run_dataset_triad_inference
    from app.database import SessionLocal
    dataset = Dataset(name='Streaming pilot', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    samples = [Sample(dataset_id=dataset.id, file_name=f'{n}.jpg', task_type='BBOX_2D', media_path=f'{n}.jpg') for n in range(2)]
    db_session.add_all(samples); db_session.commit()
    monkeypatch.setattr('app.services.triad_detector.os.path.isfile', lambda _: True)
    seen = []
    def detect(path, **kwargs):
        if seen:
            with SessionLocal() as observer:
                first = observer.get(Sample, samples[0].id)
                assert first.predictions and first.qc_issues
        seen.append(path)
        return {m: [{'label': 'car', 'confidence': .9, 'geometry': BOX}] for m in ('yolo26n', 'rtdetr_l')}
    monkeypatch.setattr('app.services.triad_detector.detect_triad_frame', detect)
    result = run_dataset_triad_inference(db_session, dataset.id, models=['yolo26n', 'rtdetr_l'])
    assert result['frames_processed'] == 2
