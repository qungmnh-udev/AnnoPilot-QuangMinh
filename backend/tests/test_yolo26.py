from pathlib import Path
import pytest
from app.models import Dataset, Sample, Annotation, Prediction
from app.services.yolo_detector import get_yolo_weights_path, resolve_yolo_model, activate_yolo_state
from app.services.qc_engine import audit_single_frame, has_yolo_predictions

ROOT = Path(__file__).resolve().parents[2]
BOX = dict(x1=10., y1=10., x2=90., y2=90.)


def test_yolo26_path_is_independent_of_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_yolo_weights_path('yolo26s') == ROOT / 'backend' / 'models' / 'yolo26s.pt'
    with pytest.raises(ValueError):
        resolve_yolo_model('yolov8n')


def test_selecting_a_variant_retires_other_yolo_sources_without_relabeling_history():
    previous = {'yolov8n': {'status': 'IMPORTED'}, 'yolo26n': {'status': 'SUCCESS'}}
    state = activate_yolo_state(previous, 'yolo26s', {'status': 'SUCCESS'})
    assert state['yolo26s']['active'] is True
    assert state['yolo26n']['active'] is False
    assert state['yolov8n']['active'] is False
    assert 'active' not in previous['yolov8n']


def test_switching_variants_preserves_history_and_uses_only_selected_yolo(client, db_session, monkeypatch):
    dataset = Dataset(name='YOLO switch', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name='frame_00.jpg', media_path=str(ROOT / 'demo_data/images/frame_00.jpg'),
                    task_type='BBOX_2D', inference_state={'yolov8n': {'status': 'IMPORTED'}})
    db_session.add(sample); db_session.flush()
    db_session.add(Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D', geometry=BOX))
    old = Prediction(sample_id=sample.id, source_model='yolov8n', label='truck', geometry=BOX, confidence=.95)
    db_session.add(old)
    db_session.add(Prediction(sample_id=sample.id, source_model='independent', label='car', geometry=BOX, confidence=.9))
    db_session.commit()
    old_id = old.id
    def fake_inference(path, **kwargs):
        return [{'label': 'car', 'confidence': .9, 'geometry': BOX}]
    monkeypatch.setattr('app.services.qc_engine.detect_image_objects', fake_inference)
    for variant in ('yolo26n', 'yolo26s'):
        summary = audit_single_frame(db_session, sample.id, model_name=variant)
        assert summary['total_issues'] == 0
    assert db_session.get(Prediction, old_id).source_model == 'yolov8n'
    assert sample.inference_state['yolo26s']['active'] is True
    assert not has_yolo_predictions(sample, 'yolo26n')  # switch back must repredict/reactivate
    sources = client.get(f'/api/datasets/{dataset.id}/ensemble/models').json()
    assert set(sources['active_models']) == {'yolo26s', 'independent'}
    result = client.post(f'/api/datasets/{dataset.id}/ensemble/run', json={'run_audit': False})
    assert result.status_code == 200
    assert set(result.json()['models_used']) == {'yolo26s', 'independent'}


def test_two_yolo26_variants_are_not_default_ensemble_members(client):
    response = client.post('/api/datasets/999999/ensemble/run-triad', json={'models': ['yolo26n', 'yolo26s']})
    assert response.status_code == 422


@pytest.mark.skipif(not get_yolo_weights_path('yolo26s').is_file(), reason='Official checkpoint is downloaded separately')
def test_actual_yolo26s_inference_through_qc_api(client, db_session, monkeypatch, tmp_path):
    from PIL import Image
    monkeypatch.setenv('YOLO_CONFIG_DIR', str(tmp_path / 'ultralytics'))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    monkeypatch.setenv('ANNO_INFERENCE_DEVICE', 'cpu')
    image_path = ROOT / 'demo_data/images/frame_00.jpg'
    with Image.open(image_path) as image:
        width, height = image.size
    dataset = Dataset(name='Real YOLO26s smoke', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset); db_session.flush()
    sample = Sample(dataset_id=dataset.id, file_name=image_path.name, media_path=str(image_path),
                    width=width, height=height, task_type='BBOX_2D')
    db_session.add(sample); db_session.commit()
    response = client.post(f'/api/datasets/{dataset.id}/qc/frame/{sample.id}/audit?model_name=yolo26s&force_repredict=true')
    assert response.status_code == 200, response.text
    assert response.json()['predictions_count'] > 0
    detail = client.get(f'/api/samples/{sample.id}').json()
    assert {p['source_model'] for p in detail['predictions']} == {'yolo26s'}
    assert detail['inference_state']['yolo26s']['device'] == 'cpu'
    assert detail['inference_state']['yolo26s']['checkpoint'].endswith('yolo26s.pt')
    for prediction in detail['predictions']:
        g = prediction['geometry']
        assert 0 <= g['x1'] < g['x2'] <= width
        assert 0 <= g['y1'] < g['y2'] <= height
