import json
import pytest
from app.models import Dataset, Sample, Annotation, Prediction, QCIssue
from app.services.qc_engine import push_dataset_annotations_to_cvat, verify_and_sync_sample_from_cvat, fetch_annotations_from_cvat_api

BOX = dict(x1=10., y1=10., x2=90., y2=90.)


class Response:
    status = 200
    def __init__(self, data):
        self.data = data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps(self.data).encode()


def test_merged_push_excludes_unreviewed_false_positive_and_duplicate_accepted_boxes(db_session, monkeypatch):
    dataset = Dataset(name='Safe merged push', format='BDD100K', task_type='BBOX_2D', cvat_task_id=1, cvat_job_id=1, cvat_token='test')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='frame_000000.jpg', frame_number=0, task_type='BBOX_2D')
    db_session.add(sample); db_session.flush()
    db_session.add(Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D', geometry=BOX))
    for status in ('PENDING', 'FALSE_POSITIVE', 'ACCEPTED'):
        db_session.add(QCIssue(sample_id=sample.id, issue_type='MISSING_OBJECT', suggested_label='car',
                              location=BOX if status == 'ACCEPTED' else dict(x1=100, y1=100, x2=200, y2=200),
                              qc_score=.9, status=status))
    db_session.commit()
    pushed = []
    def urlopen(request, **kwargs):
        if '/api/labels?' in request.full_url:
            return Response({'results': [{'id': 7, 'name': 'car'}]})
        if request.get_method() == 'GET':
            return Response({'shapes': []})
        pushed.extend(json.loads(request.data)['shapes'])
        return Response({})
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    result = push_dataset_annotations_to_cvat(db_session, dataset.id)
    assert result['pushed_shapes_count'] == 1
    assert pushed[0]['points'] == [10., 10., 90., 90.]


def test_append_retry_does_not_resend_existing_shape(db_session, monkeypatch):
    dataset = Dataset(name='Append retry', format='BDD100K', task_type='BBOX_2D', cvat_task_id=1, cvat_job_id=1, cvat_token='test')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='frame_000000.jpg', frame_number=0, task_type='BBOX_2D')
    db_session.add(sample); db_session.flush()
    db_session.add(Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D', geometry=BOX))
    db_session.commit()
    def urlopen(request, **kwargs):
        assert request.get_method() == 'GET'
        if '/api/labels?' in request.full_url:
            return Response({'results': [{'id': 7, 'name': 'car'}]})
        return Response({'shapes': [{'type': 'rectangle', 'frame': 0, 'label_id': 7, 'points': [10., 10., 90., 90.]}]})
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    assert push_dataset_annotations_to_cvat(db_session, dataset.id)['pushed_shapes_count'] == 0


def test_live_verification_preserves_resolved_issue_and_history(db_session, monkeypatch):
    dataset = Dataset(name='Live verification', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='frame_000000.jpg', frame_number=0, task_type='BBOX_2D')
    db_session.add(sample); db_session.flush()
    db_session.add(Prediction(sample_id=sample.id, source_model='a', label='car', geometry=BOX, confidence=.9))
    db_session.add(QCIssue(sample_id=sample.id, issue_type='MISSING_OBJECT', suggested_label='car', location=BOX, qc_score=.9, status='PENDING'))
    db_session.commit()
    monkeypatch.setattr('app.services.qc_engine.fetch_annotations_from_cvat_api', lambda _: {0: [{'label': 'car', 'geometry': BOX, 'shape_type': 'BBOX_2D'}]})
    result = verify_and_sync_sample_from_cvat(db_session, sample.id)
    assert result['resolved_issues_count'] == 1
    assert len(sample.qc_issues) == 1 and sample.qc_issues[0].status == 'RESOLVED'
    assert len(sample.review_events) == 1


def test_live_rectangle_track_is_interpolated_between_keyframes(db_session, monkeypatch):
    dataset = Dataset(name='Track frames', format='CVAT', task_type='BBOX_2D', cvat_task_id=1, cvat_token='test')
    db_session.add(dataset); db_session.flush()
    db_session.add(Sample(dataset_id=dataset.id, file_name='frame_000001.jpg', frame_number=1, task_type='BBOX_2D'))
    db_session.commit()
    def urlopen(request, **kwargs):
        if '/api/labels?' in request.full_url:
            return Response({'results': [{'id': 7, 'name': 'car'}]})
        return Response({'tracks': [{'id': 3, 'label_id': 7, 'shapes': [
            {'frame': 0, 'type': 'rectangle', 'points': [0, 0, 40, 40], 'outside': False},
            {'frame': 2, 'type': 'rectangle', 'points': [20, 20, 60, 60], 'outside': False},
        ]}]})
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    frames = fetch_annotations_from_cvat_api(dataset)
    assert frames[1][0]['geometry'] == dict(x1=10., y1=10., x2=50., y2=50.)


def test_cvat_image_id_is_preserved_instead_of_filename_digits(client):
    xml = b'<annotations><image id="4" name="bdd12345.jpg" width="100" height="100"><box label="car" xtl="0" ytl="0" xbr="40" ybr="40" /></image></annotations>'
    response = client.post('/api/datasets/upload', data={'name': 'Frame metadata', 'format': 'CVAT'}, files={'annotations': ('labels.xml', xml, 'application/xml')})
    assert response.status_code == 200
    samples = client.get(f"/api/datasets/{response.json()['id']}/samples").json()
    assert samples[0]['frame_number'] == 4
