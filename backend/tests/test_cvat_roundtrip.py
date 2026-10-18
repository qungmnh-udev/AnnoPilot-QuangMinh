import copy
import json
from urllib.parse import urlsplit, parse_qs
import pytest

from app.models import Dataset, Sample, Annotation, Prediction
from app.services.qc_engine import push_dataset_annotations_to_cvat, sync_cvat_dataset_annotations
from app.parsers import CVATParser

BOX = dict(x1=10., y1=10., x2=90., y2=90.)


class Response:
    status = 200
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.body).encode()


class RemoteCVAT:
    """Stateful protocol fixture: observes exactly which shapes the client creates/replaces."""
    def __init__(self):
        self.frames = [{'name': 'a.jpg', 'width': 100, 'height': 100}, {'name': 'b.jpg', 'width': 100, 'height': 100}]
        self.annotations = {'version': 1, 'shapes': [], 'tracks': [], 'tags': []}
        self.writes = []
        self.urls = []
        self.job = {'task_id': 1, 'type': 'annotation', 'start_frame': 0, 'stop_frame': 1}

    def __call__(self, request, **kwargs):
        self.urls.append(request.full_url)
        path = urlsplit(request.full_url).path
        query = parse_qs(urlsplit(request.full_url).query)
        method = request.get_method()
        if path == '/api/jobs/5': return Response(self.job)
        if path == '/api/tasks/1/data/meta': return Response({'size': 2, 'frames': self.frames})
        if path == '/api/labels':
            if query.get('page') == ['2']:
                return Response({'results': [{'id': 8, 'name': 'bus', 'attributes': [{'id': 81, 'name': 'review_note'}]}], 'next': None})
            return Response({'results': [{'id': 7, 'name': 'car', 'attributes': [{'id': 71, 'name': 'review_note'}]}], 'next': '/api/labels?task_id=1&page=2'})
        assert path in ('/api/tasks/1/annotations', '/api/jobs/5/annotations'), request.full_url
        if method == 'GET': return Response(copy.deepcopy(self.annotations))
        payload = json.loads(request.data)
        self.writes.append((method, payload))
        if method == 'PATCH':
            assert query['action'] == ['create']
            for shape in payload['shapes']:
                self.annotations['shapes'].append(dict(shape, id=100 + len(self.annotations['shapes'])))
        elif method == 'PUT': self.annotations = payload
        else: raise AssertionError(method)
        return Response(self.annotations)


def setup_dataset(db, monkeypatch, label='car'):
    remote = RemoteCVAT()
    monkeypatch.setattr('urllib.request.urlopen', remote)
    dataset = Dataset(name='Round trip', format='CVAT', task_type='BBOX_2D', cvat_task_id=1, cvat_token='test')
    db.add(dataset); db.flush()
    sample = Sample(dataset_id=dataset.id, file_name='a.jpg', frame_number=99, task_type='BBOX_2D', width=100, height=100)
    db.add(sample); db.flush()
    db.add(Annotation(sample_id=sample.id, label=label, shape_type='BBOX_2D', geometry=BOX, attributes={'review_note': 'reviewed'}))
    db.commit()
    return dataset, sample, remote


def test_roundtrip_maps_names_not_guessed_numbers_preserves_attributes_and_retry(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    result = push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='annotations')
    assert result['cvat_job_id'] is None  # Full task is never silently redirected to its first job.
    assert remote.annotations['shapes'][0]['frame'] == 0
    assert remote.annotations['shapes'][0]['attributes'] == [{'spec_id': 71, 'value': 'reviewed'}]
    assert sample.frame_number == 0
    # A reviewer edits the shape on CVAT, changing both class and coordinates.
    remote.annotations['shapes'][0].update(label_id=8, points=[10, 10, 95, 90], attributes=[{'spec_id': 81, 'value': 'fixed in CVAT'}])
    sync_cvat_dataset_annotations(db_session, dataset.id)
    db_session.refresh(sample)
    assert len(sample.annotations) == 1
    annotation = sample.annotations[0]
    assert annotation.label == 'bus' and annotation.geometry['x2'] == 95
    assert annotation.attributes == {'review_note': 'fixed in CVAT'}
    assert annotation.source_metadata['cvat_shape_id'] == 100
    assert push_dataset_annotations_to_cvat(db_session, dataset.id)['pushed_shapes_count'] == 0
    assert len(remote.writes) == 1


@pytest.mark.parametrize('remote_label', ['motorcycle', 'motorbike'])
def test_motorcycle_push_uses_matching_cvat_label_id(db_session, monkeypatch, remote_label):
    from app.services.cvat_api import CVATClient
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    monkeypatch.setattr(CVATClient, 'labels', lambda self: [
        {'id': 7, 'name': 'car'}, {'id': 42, 'name': remote_label},
    ])
    db_session.add(Prediction(sample_id=sample.id, source_model='yolo26n',
                              label='motorcycle', confidence=.9, geometry=BOX))
    db_session.commit()
    result = push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='predictions')
    assert result['pushed_shapes_count'] == 1
    assert remote.annotations['shapes'][0]['label_id'] == 42


def test_missing_motorcycle_class_reports_task_and_labels_without_writing(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    db_session.add(Prediction(sample_id=sample.id, source_model='yolo26n',
                              label='motorcycle', confidence=.9, geometry=BOX))
    db_session.commit()
    with pytest.raises(ValueError, match="no matching class for 'motorcycle'") as error:
        push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='predictions')
    assert 'Task #1' in str(error.value)
    assert 'Available CVAT classes: bus, car' in str(error.value)
    assert 'Project' in str(error.value)
    assert not remote.writes
    assert not remote.annotations['shapes']


def test_xml_without_declaration_syncs_and_unmatched_name_cannot_overwrite_by_frame_id(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    good = b'<annotations><image id="4" name="a.jpg" width="100" height="100"><box label="bus" xtl="10" ytl="10" xbr="90" ybr="90" /></image></annotations>'
    assert sync_cvat_dataset_annotations(db_session, dataset.id, content=good)['updated_frames_count'] == 1
    db_session.refresh(sample)
    assert sample.annotations[0].label == 'bus' and sample.frame_number == 4
    bad = good.replace(b'a.jpg', b'other.jpg').replace(b'bus', b'car')
    with pytest.raises(ValueError, match='No exported image names'):
        sync_cvat_dataset_annotations(db_session, dataset.id, content=bad)
    assert sample.annotations[0].label == 'bus'
    assert not remote.urls  # File sync did not pretend to fetch live annotations.


def test_empty_file_is_not_a_live_sync_fallback(client, monkeypatch):
    from app.database import SessionLocal
    with SessionLocal() as db:
        dataset, _, remote = setup_dataset(db, monkeypatch)
        dataset_id = dataset.id
    result = client.post(f'/api/datasets/{dataset_id}/qc/cvat-sync', files={'file': ('empty.xml', b'', 'application/xml')})
    assert result.status_code == 422
    assert not remote.urls


@pytest.mark.parametrize('mutation', ['missing_name', 'wrong_size', 'wrong_job', 'outside_job'])
def test_mapping_failure_precedes_any_local_or_remote_mutation(db_session, monkeypatch, mutation):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    if mutation == 'missing_name': remote.frames[0]['name'] = 'unrelated.jpg'
    if mutation == 'wrong_size': remote.frames[0]['width'] = 101
    if mutation in ('wrong_job', 'outside_job'):
        dataset.cvat_job_id = 5
        if mutation == 'wrong_job': remote.job['task_id'] = 2
        else: remote.job['start_frame'] = 1
        db_session.commit()
    with pytest.raises(ValueError): sync_cvat_dataset_annotations(db_session, dataset.id)
    with pytest.raises(ValueError): push_dataset_annotations_to_cvat(db_session, dataset.id)
    assert sample.annotations[0].label == 'car' and sample.frame_number == 99
    assert not remote.writes


def test_wrong_class_append_rejected_instead_of_creating_second_box(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch, label='bus')
    sample.annotations[0].attributes = {'previous_label': 'car'}
    db_session.commit()
    remote.annotations['shapes'] = [{'id': 9, 'type': 'rectangle', 'frame': 0, 'label_id': 7, 'points': [10, 10, 90, 90]}]
    with pytest.raises(ValueError, match='Append cannot replace'):
        push_dataset_annotations_to_cvat(db_session, dataset.id)
    assert not remote.writes


def test_partial_replace_preserves_other_frames_and_non_bbox_shapes(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    outside = {'id': 21, 'type': 'rectangle', 'frame': 1, 'label_id': 7, 'points': [1, 1, 50, 50]}
    polygon = {'id': 22, 'type': 'polygon', 'frame': 0, 'label_id': 7, 'points': [1, 1, 50, 1, 50, 50]}
    remote.annotations.update(shapes=[outside, polygon], tags=[{'id': 30, 'frame': 1, 'label_id': 7}])
    push_dataset_annotations_to_cvat(db_session, dataset.id, mode='replace')
    payload = remote.writes[0][1]
    assert outside in payload['shapes'] and polygon in payload['shapes']
    assert payload['tags'] == [{'id': 30, 'frame': 1, 'label_id': 7}]
    assert payload['version'] == 1


def test_predictions_do_not_export_inactive_or_unfused_multiple_models(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    sample.inference_state = {'yolov8n': {'active': False}, 'yolo26n': {'active': True}}
    db_session.add_all([Prediction(sample_id=sample.id, source_model=m, label='car', confidence=.9, geometry=BOX)
                        for m in ('yolov8n', 'yolo26n')])
    db_session.commit()
    assert push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='predictions')['pushed_shapes_count'] == 1
    db_session.add(Prediction(sample_id=sample.id, source_model='rtdetr_l', label='car', confidence=.9, geometry=BOX))
    db_session.commit()
    with pytest.raises(ValueError, match='need WBF'):
        push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='predictions')
    sample.inference_state = {**sample.inference_state, 'rtdetr_l': {'status': 'ERROR'}}
    db_session.commit()
    assert push_dataset_annotations_to_cvat(db_session, dataset.id, target_source='predictions')['pushed_shapes_count'] == 0


def test_job_id_from_explicit_url_not_segment_id(tmp_path):
    path = tmp_path / 'export.xml'
    path.write_text('<annotations><meta><task><id>1</id><segments><segment><id>99</id><url>https://cvat.example/tasks/1/jobs/5</url></segment></segments></task></meta><image id="0" name="a.jpg" /></annotations>')
    parser = CVATParser()
    parser.parse([path], [])
    assert parser.cvat_job_id == 5 and parser.cvat_task_id == 1


def test_can_clear_job_scope_and_partial_config_does_not_reset_url(client):
    from app.database import SessionLocal
    with SessionLocal() as db:
        dataset = Dataset(name='Config', format='CVAT', task_type='BBOX_2D', cvat_task_id=1,
                          cvat_job_id=5, cvat_base_url='https://cvat.example', cvat_token='test')
        db.add(dataset); db.commit(); dataset_id = dataset.id
    result = client.put(f'/api/datasets/{dataset_id}/cvat-config', json={'cvat_job_id': None})
    assert result.status_code == 200
    assert result.json()['cvat_job_id'] is None
    assert result.json()['cvat_base_url'] == 'https://cvat.example'


def test_empty_replace_clears_only_explicit_selected_frame(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    remote.annotations['shapes'] = [
        {'id': 1, 'type': 'rectangle', 'frame': 0, 'label_id': 7, 'points': [10, 10, 90, 90]},
        {'id': 2, 'type': 'rectangle', 'frame': 1, 'label_id': 7, 'points': [10, 10, 90, 90]},
    ]
    for annotation in list(sample.annotations): db_session.delete(annotation)
    db_session.commit(); db_session.expire(sample, ['annotations'])
    push_dataset_annotations_to_cvat(db_session, dataset.id, mode='replace')
    assert [s['frame'] for s in remote.annotations['shapes']] == [1]


def test_replace_refuses_tracks_instead_of_duplicating_interpolated_boxes(db_session, monkeypatch):
    dataset, sample, remote = setup_dataset(db_session, monkeypatch)
    remote.annotations['tracks'] = [{'id': 3, 'label_id': 7, 'shapes': []}]
    with pytest.raises(ValueError, match='video tracks'):
        push_dataset_annotations_to_cvat(db_session, dataset.id, mode='replace')
    assert not remote.writes
