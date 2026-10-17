"""Isolated real training/API/QC smoke. Synthetic images are not quality evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
scratch = ROOT / '.runtime' / ('training-smoke-' + str(uuid.uuid4()))
scratch.mkdir(parents=True)
os.environ['DATABASE_URL'] = 'sqlite:///' + str(scratch / 'smoke.db')
os.environ['UPLOAD_DIR'] = str(scratch / 'uploads')
os.environ['EXPORT_DIR'] = str(scratch / 'exports')
os.environ['TRAINING_DIR'] = str(scratch / 'jobs')
os.environ['ANNO_INFERENCE_DEVICE'] = 'cpu'
sys.path.insert(0, str(ROOT / 'backend'))


def main():
    from PIL import Image, ImageDraw
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import SessionLocal
    from app.models import Annotation, Dataset, Sample
    from app.services.training import job_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['yolo26n'])
    args = parser.parse_args()
    with SessionLocal() as db:
        dataset = Dataset(name='SYNTHETIC training smoke - not quality evidence', format='CVAT', task_type='BBOX_2D')
        db.add(dataset); db.flush()
        for index in range(4):
            path = scratch / f'{index}.png'
            image = Image.new('RGB', (128, 128), (index * 40, 30, 50))
            ImageDraw.Draw(image).rectangle((20, 40, 90, 100), fill=(200, 200, 200))
            image.save(path)
            sample = Sample(dataset_id=dataset.id, file_name=path.name, media_path=str(path),
                            width=128, height=128, task_type='BBOX_2D')
            db.add(sample); db.flush()
            db.add(Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D',
                              geometry=dict(x1=20, y1=40, x2=90, y2=100)))
        db.commit()
        dataset_id = dataset.id
    client = TestClient(app)
    results = []
    print(f'Artifacts: {scratch}', flush=True)
    for model in args.models:
        response = client.post(f'/api/datasets/{dataset_id}/training', json=dict(
            model_name=model, epochs=1, image_size=320, batch_size=1, device='cpu', labels_reviewed=True))
        assert response.status_code == 202, response.text
        job_id = response.json()['id']
        print(f'{model}: job {job_id}', flush=True)
        deadline = time.monotonic() + 900
        while True:
            history = client.get(f'/api/datasets/{dataset_id}/training').json()
            job = next(j for j in history['jobs'] if j['id'] == job_id)
            if job['status'] not in ('QUEUED', 'PREPARING', 'TRAINING', 'VALIDATING', 'REGISTERING'):
                break
            if time.monotonic() > deadline:
                client.post(f'/api/datasets/{dataset_id}/training/{job_id}/cancel')
                raise RuntimeError('Smoke timeout; cancellation requested')
            time.sleep(2)
        if job['status'] != 'COMPLETED':
            print((job_dir(job_id) / 'training.log').read_text(encoding='utf-8', errors='replace')[-7000:])
        assert job['status'] == 'COMPLETED', job
        checkpoint = next(c for c in history['checkpoints'] if c['job_id'] == job_id)
        response = client.post(f'/api/datasets/{dataset_id}/ensemble/run-triad', json=dict(
            models=[model], auto_ensemble=False, sample_limit=1))
        assert response.status_code == 200, response.text
        with SessionLocal() as db:
            sample = db.query(Sample).filter_by(dataset_id=dataset_id).first()
            state = sample.inference_state[model]
            assert state['training_job_id'] == job_id
            assert state['checkpoint_sha256'] == checkpoint['sha256']
        results.append(dict(model=model, job=job, checkpoint=checkpoint, qc=response.json()))
        print(f'{model}: real training completed; checkpoint automatically used by QC.', flush=True)
        (scratch / 'smoke.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('PASS: real training -> checkpoint registry -> real QC inference. Synthetic smoke only.', flush=True)


if __name__ == '__main__':
    main()
