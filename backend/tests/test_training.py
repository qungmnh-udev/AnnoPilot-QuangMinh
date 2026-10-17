import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
import pytest
from sqlalchemy import select

from app.models import Annotation, Dataset, GoldenBenchmark, ModelCheckpoint, Prediction, Sample, TrainingJob
from app.schemas.training import TrainingRequest
from app.services import training, training_worker
from app.services.model_registry import active_checkpoint, checkpoint_scope
from app.services.yolo_detector import get_yolo_weights_path


def make_dataset(db, tmp_path, count=7):
    dataset = Dataset(name='Training fixture', format='CVAT', task_type='BBOX_2D')
    db.add(dataset); db.flush()
    for index in range(count):
        path = tmp_path / f'image-{index}.png'
        Image.new('RGB', (64, 64), (index * 30, 20, 40)).save(path)
        sample = Sample(dataset_id=dataset.id, file_name=path.name, media_path=str(path),
                        task_type='BBOX_2D', width=64, height=64)
        db.add(sample); db.flush()
        db.add(Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D',
                          geometry=dict(x1=5, y1=5, x2=40, y2=40)))
    db.commit()
    return dataset


@pytest.fixture
def worker_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('TRAINING_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr(training, 'launch_worker', Mock())
    monkeypatch.setattr(training, 'select_device', lambda requested: 'cpu')


def test_snapshot_excludes_golden_and_duplicates_and_exports_separate_splits(db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    samples = db_session.scalars(select(Sample).where(Sample.dataset_id == dataset.id)).all()
    db_session.add(GoldenBenchmark(dataset_id=dataset.id, verified_by='Reviewer', samples=[
        {'file_name': samples[0].file_name, 'annotations': []}]))
    duplicate = Sample(dataset_id=dataset.id, file_name='duplicate.png', media_path=samples[1].media_path,
                       task_type='BBOX_2D', width=64, height=64)
    db_session.add(duplicate); db_session.flush()
    db_session.add(Annotation(sample_id=duplicate.id, label='car', shape_type='BBOX_2D',
                              geometry=dict(x1=5, y1=5, x2=40, y2=40)))
    db_session.commit()
    frames, counts = training.collect_snapshot(db_session, dataset)
    assert counts['golden'] == counts['duplicates'] == 1
    assert counts['eligible'] == 6
    train, val = training.split_frames(frames, .2, 42)
    assert not {f['sha256'] for f in train} & {f['sha256'] for f in val}
    assert (train, val) == training.split_frames(frames, .2, 42)
    manifest = dict(train=train, val=val, classes=training.CLASSES)
    folder = tmp_path / 'snapshot'
    progress = training_worker.Progress(folder, 1)
    training_worker.prepare_data(manifest, folder, progress)
    config = json.loads((folder / 'data.yaml').read_text())
    assert config['train'] != config['val']
    assert len(list((folder / 'data/images/train').glob('*.png'))) == len(train)
    assert 'car' in config['names'].values()
    label_file = next((folder / 'data/labels/train').glob('*.txt'))
    values = label_file.read_text().split()
    assert values[0] == '2' and all(0 <= float(v) <= 1 for v in values[1:])


def test_golden_copy_in_another_dataset_is_excluded(db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    other = Dataset(name='Independent Golden', format='CVAT', task_type='BBOX_2D')
    db_session.add(other); db_session.flush()
    original = db_session.scalar(select(Sample).where(Sample.dataset_id == dataset.id))
    db_session.add(Sample(dataset_id=other.id, file_name='renamed.png', media_path=original.media_path, task_type='BBOX_2D'))
    db_session.add(GoldenBenchmark(dataset_id=other.id, verified_by='Reviewer', samples=[{'file_name': 'renamed.png', 'annotations': []}]))
    db_session.commit()
    frames, counts = training.collect_snapshot(db_session, dataset)
    assert counts['golden'] == 1
    assert original.id not in [f['sample_id'] for f in frames]


def test_start_cancel_history_and_global_training_lock(client, db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    endpoint = f'/api/datasets/{dataset.id}/training'
    assert client.get(endpoint + '/preview').json()['ready']
    assert client.post(endpoint, json={'model_name': 'yolo26n'}).status_code == 422
    assert client.post(endpoint, json={'model_name': 'yolov8n', 'labels_reviewed': True}).status_code == 422
    response = client.post(endpoint, json={'labels_reviewed': True, 'epochs': 1})
    assert response.status_code == 202, response.text
    job = response.json()
    assert job['config']['train_count'] + job['config']['validation_count'] == 7
    training.launch_worker.assert_called_once_with(job['id'])
    assert client.post(endpoint, json={'labels_reviewed': True}).status_code == 409
    assert client.get(endpoint).json()['jobs'][0]['id'] == job['id']
    assert client.post(endpoint + f"/{job['id']}/cancel").status_code == 200
    assert (training.job_dir(job['id']) / 'cancel').is_file()
    assert client.get(f"/api/datasets/9999/training/{job['id']}/log").status_code == 404


def test_worker_registers_checkpoint_and_qc_uses_it_without_manual_copy(db_session, tmp_path, worker_environment, monkeypatch):
    dataset = make_dataset(db_session, tmp_path)
    sample = db_session.scalar(select(Sample).where(Sample.dataset_id == dataset.id))
    sample.inference_state = {'yolo26n': {'status': 'SUCCESS', 'active': True}}
    db_session.commit()
    job = training.start_training(db_session, dataset, TrainingRequest(labels_reviewed=True, epochs=1))

    def train(manifest, folder, progress):
        assert len(manifest['train']) > len(manifest['val'])
        path = folder / 'best.pt'
        path.write_bytes(b'validated checkpoint fixture')
        progress.update(epoch=1)
        return path, {'metrics/mAP50(B)': .5}

    monkeypatch.setattr(training_worker, 'train_ultralytics', train)
    training_worker.run_job(job['id'])
    db_session.expire_all()
    row = db_session.scalar(select(ModelCheckpoint))
    assert row.active and row.dataset_id == dataset.id
    assert db_session.get(TrainingJob, job['id']).status == 'COMPLETED'
    assert db_session.get(TrainingJob, job['id']).active_slot is None
    assert sample.inference_state['yolo26n']['status'] == 'STALE'
    assert active_checkpoint('yolo26n', dataset.id)['job_id'] == job['id']
    with checkpoint_scope(dataset.id):
        assert get_yolo_weights_path('yolo26n') == Path(row.path)
    assert get_yolo_weights_path('yolo26n') != Path(row.path)
    training.activate_checkpoint(db_session, dataset.id, model_name='yolo26n')
    assert active_checkpoint('yolo26n', dataset.id) is None
    training.activate_checkpoint(db_session, dataset.id, checkpoint_id=row.id)
    assert active_checkpoint('yolo26n', dataset.id)['id'] == row.id
    from app.services.qc_engine import audit_single_frame
    calls = []
    def inference(path, **kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr('app.services.qc_engine.detect_image_objects', inference)
    audit_single_frame(db_session, sample.id, run_detector=True, model_name='yolo26n')
    assert calls[0]['model_path'] == Path(row.path)
    assert sample.inference_state['yolo26n']['training_job_id'] == job['id']
    assert sample.inference_state['yolo26n']['checkpoint_sha256'] == row.sha256


@pytest.mark.parametrize('cancel', [False, True])
def test_failed_or_cancelled_worker_never_activates_partial_weights(db_session, tmp_path, worker_environment, monkeypatch, cancel):
    dataset = make_dataset(db_session, tmp_path)
    job = training.start_training(db_session, dataset, TrainingRequest(labels_reviewed=True, epochs=1))
    def fail(*args):
        raise RuntimeError('training failed')
    monkeypatch.setattr(training_worker, 'train_ultralytics', fail)
    if cancel:
        (training.job_dir(job['id']) / 'cancel').touch()
    training_worker.run_job(job['id'])
    db_session.expire_all()
    assert db_session.get(TrainingJob, job['id']).status == ('CANCELLED' if cancel else 'FAILED')
    assert db_session.scalar(select(ModelCheckpoint)) is None
    assert db_session.get(TrainingJob, job['id']).active_slot is None


def test_dead_worker_releases_persistent_slot(db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    job = training.start_training(db_session, dataset, TrainingRequest(labels_reviewed=True))
    row = db_session.get(TrainingJob, job['id'])
    row.progress = {**row.progress, 'updated_at': time.time() - 300}
    db_session.commit()
    assert training.job_view(db_session, row)['status'] == 'INTERRUPTED'
    assert row.active_slot is None


def test_snapshot_rejects_bad_boxes_and_changed_images(db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    frames, _ = training.collect_snapshot(db_session, dataset)
    train, val = training.split_frames(frames, .2, 1)
    Path(train[0]['source_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='train'):
        training_worker.prepare_data(dict(train=train, val=val), tmp_path / 'copy', training_worker.Progress(tmp_path / 'copy', 1))


def test_gpu_request_never_silently_uses_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='unavailable'):
        training.select_device('gpu')
    assert training.select_device('auto') == 'cpu'


def test_legacy_predictions_cannot_be_imported(client, db_session, tmp_path):
    dataset = make_dataset(db_session, tmp_path)
    response = client.post(f'/api/datasets/{dataset.id}/qc/predictions', json={
        'source_model': 'yolov8n', 'predictions': []})
    assert response.status_code == 422


def test_gpu_device_is_forwarded_to_rtdetr(monkeypatch):
    import ultralytics
    from app.services import triad_detector
    model = Mock()
    model.predict.return_value = []
    monkeypatch.setattr(ultralytics, 'RTDETR', lambda path: model)
    monkeypatch.setattr(triad_detector, 'inference_device', lambda *args: 'cuda:0')
    monkeypatch.setattr(triad_detector, '_MODEL_CACHE', {})
    assert triad_detector.run_rtdetr_inference('fixture.jpg', weights_path='fixture.pt') == []
    assert model.predict.call_args.kwargs['device'] == 'cuda:0'


def test_gpu_device_moves_fasterrcnn_and_input(monkeypatch):
    import torch
    import torchvision.models.detection as detection
    import torchvision.transforms.functional as transforms
    from app.services import triad_detector
    model = Mock()
    model.to.return_value.eval.return_value = None
    model.return_value = [dict(scores=torch.tensor([]), labels=torch.tensor([], dtype=torch.int64), boxes=torch.empty((0, 4)))]
    tensor = Mock()
    tensor.unsqueeze.return_value.to.return_value = 'gpu-input'
    monkeypatch.setattr(detection, 'fasterrcnn_mobilenet_v3_large_fpn', lambda **kwargs: model)
    monkeypatch.setattr(transforms, 'to_tensor', lambda image: tensor)
    monkeypatch.setattr(triad_detector.Image, 'open', lambda path: Mock())
    monkeypatch.setattr(triad_detector, 'inference_device', lambda *args: 'cuda:0')
    monkeypatch.setattr(triad_detector, '_MODEL_CACHE', {})
    assert triad_detector.run_fasterrcnn_inference('fixture.jpg') == []
    model.to.assert_called_once_with('cuda:0')
    tensor.unsqueeze.return_value.to.assert_called_once_with('cuda:0')
    model.assert_called_once_with('gpu-input')


def test_golden_from_training_images_is_rejected_by_wbf_experiment(client, db_session, tmp_path, worker_environment):
    dataset = make_dataset(db_session, tmp_path)
    sample = db_session.scalar(select(Sample).where(Sample.dataset_id == dataset.id))
    job_id = str(uuid.uuid4())
    db_session.add(TrainingJob(id=job_id, dataset_id=dataset.id, model_name='yolo26n', status='COMPLETED', config={}))
    db_session.flush()
    db_session.add(ModelCheckpoint(dataset_id=dataset.id, job_id=job_id, model_name='yolo26n',
                                   path='fixture.pt', sha256='fixture', active=True,
                                   training_images=[training.file_hash(sample.media_path)]))
    benchmark = GoldenBenchmark(dataset_id=dataset.id, verified_by='Reviewer',
                                samples=[dict(file_name=sample.file_name, annotations=[])])
    db_session.add(benchmark); db_session.commit()
    response = client.post(f'/api/datasets/{dataset.id}/ensemble/experiments', json={'benchmark_id': benchmark.id})
    assert response.status_code == 422
    assert 'independent' in response.json()['detail']
