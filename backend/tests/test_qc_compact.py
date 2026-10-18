import json

from sqlalchemy import event

from app.database import engine
from app.models import Annotation, Dataset, Prediction, QCIssue, Sample


def test_compact_qc_avoids_bulk_evidence_and_preserves_detail(client, db_session):
    dataset = Dataset(name='Compact QC', format='CVAT', task_type='BBOX_2D')
    db_session.add(dataset)
    db_session.flush()
    rows = [Sample(dataset_id=dataset.id, file_name=f'{i}.jpg', task_type='BBOX_2D',
                   inference_state={'yolo26n': {'status': 'SUCCESS'}}) for i in range(3)]
    rows[2].inference_state = {'yolo26n': {'status': 'ERROR'}}
    db_session.add_all(rows)
    db_session.flush()
    geometry = dict(x1=1, y1=2, x2=20, y2=30)
    evidence = {'diagnostic': 'x' * 100_000}
    db_session.add_all([
        Annotation(sample_id=rows[0].id, label='car', shape_type='BBOX_2D', geometry=geometry),
        Prediction(sample_id=rows[0].id, label='car', source_model='yolo26n', confidence=.9,
                   geometry=geometry, extra_data=evidence),
        QCIssue(sample_id=rows[0].id, issue_type='MISSING_OBJECT', suggested_label='car',
                location=geometry, qc_score=.8, evidence=evidence, status='PENDING'),
        QCIssue(sample_id=rows[0].id, issue_type='WRONG_CLASS', suggested_label='truck',
                location=geometry, qc_score=.7, evidence=evidence, status='RESOLVED'),
    ])
    db_session.commit()
    dataset_id, sample_id = dataset.id, rows[0].id
    queries = []

    def record(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            queries.append(statement)

    event.listen(engine, 'before_cursor_execute', record)
    try:
        response = client.get(f'/api/datasets/{dataset_id}/qc/report?compact=true')
        assert response.status_code == 200
        summary = response.json()
        assert summary['total_samples'] == 3
        assert summary['flagged_samples_count'] == 1
        assert summary['clean_samples_count'] == 1
        assert summary['processed_samples_count'] == 2
        assert summary['inference_error_samples_count'] == 1
        assert summary['total_missing_candidates'] == 1
        assert summary['total_wrong_class_candidates'] == 0
        assert summary['top_suspicious_samples'] == []
        assert len(response.content) < 2000
        assert not any('qc_issues.evidence' in query or 'predictions.extra_data' in query for query in queries)
        queries.clear()

        response = client.get(f'/api/datasets/{dataset_id}/qc/queue?compact=true&include_clean=true')
        assert response.status_code == 200
        queue = response.json()
        assert len(queue) == 3
        assert queue[0]['id'] == sample_id
        assert queue[0]['qc_issue_count'] == 1
        assert queue[0]['qc_issue_counts'] == {'missing': 1, 'wrong': 1, 'resolved': 1, 'total': 2}
        assert queue[0]['annotation_count'] == 1
        assert queue[0]['prediction_count'] == 1
        assert all('predictions' not in row and 'qc_issues' not in row and 'annotations' not in row for row in queue)
        assert len(response.content) < 10000
        assert len(queries) <= 6
        assert not any('qc_issues.evidence' in query or 'predictions.extra_data' in query for query in queries)
    finally:
        event.remove(engine, 'before_cursor_execute', record)

    # Full details remain available for the single selected frame and older API clients.
    detail = client.get(f'/api/samples/{sample_id}').json()
    assert detail['qc_issues'][0]['evidence'] == evidence
    assert detail['predictions'][0]['extra_data'] == evidence
    full_queue = client.get(f'/api/datasets/{dataset_id}/qc/queue').json()
    assert full_queue[0]['qc_issues'][0]['evidence'] == evidence
    full_report = client.get(f'/api/datasets/{dataset_id}/qc/report').json()
    for key, value in summary.items():
        if key != 'top_suspicious_samples':
            assert full_report[key] == value


def test_compact_qc_empty_and_unknown_dataset(client):
    response = client.post('/api/datasets/upload', data={'name': 'Empty labels', 'format': 'BDD100K'},
                           files={'annotations': ('labels.json', json.dumps([{'name': 'empty.jpg', 'labels': []}]), 'application/json')})
    assert response.status_code == 200
    dataset_id = response.json()['id']
    summary = client.get(f'/api/datasets/{dataset_id}/qc/report?compact=true').json()
    assert summary['unprocessed_samples_count'] == 1
    assert summary['clean_samples_count'] == 0
    assert client.get(f'/api/datasets/{dataset_id}/qc/queue?compact=true').json() == []
    assert client.get('/api/datasets/999999/qc/report?compact=true').status_code == 404
    assert client.get('/api/datasets/999999/qc/queue?compact=true').status_code == 404
