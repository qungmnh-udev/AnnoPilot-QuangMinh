from sqlalchemy import event

from app.database import engine
from app.models import Annotation, Dataset, Prediction, QCIssue, Sample


def test_dataset_browsing_batches_queries_and_preserves_counts(client, db_session):
    datasets = [Dataset(name=f'Dataset {i}', format='CVAT', task_type='BBOX_2D') for i in range(3)]
    db_session.add_all(datasets)
    db_session.flush()
    geometry = dict(x1=1, y1=2, x2=20, y2=30)
    for dataset in datasets:
        for index in range(12):
            sample = Sample(dataset_id=dataset.id, file_name=f'{index}.jpg', task_type='BBOX_2D')
            db_session.add(sample)
            db_session.flush()
            db_session.add_all([
                Annotation(sample_id=sample.id, label='car', shape_type='BBOX_2D', geometry=geometry),
                Prediction(sample_id=sample.id, source_model='yolo26n', label='car', confidence=.9, geometry=geometry),
                QCIssue(sample_id=sample.id, issue_type='MISSING_OBJECT', suggested_label='car',
                        location=geometry, qc_score=.8, status='PENDING'),
            ])
    db_session.commit()
    dataset_id = datasets[0].id
    queries = []

    def record(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            queries.append(statement)

    event.listen(engine, 'before_cursor_execute', record)
    try:
        response = client.get('/api/datasets')
        assert response.status_code == 200
        assert all(row['sample_count'] == 12 and row['annotation_count'] == 12
                   and row['classes'] == {'car': 12} for row in response.json())
        assert len(queries) <= 8, f'Dataset list made {len(queries)} SELECT queries'
        queries.clear()
        response = client.get(f'/api/datasets/{dataset_id}/samples')
        assert response.status_code == 200
        assert len(response.json()) == 12
        assert all(row['annotation_count'] == 1 and row['prediction_count'] == 1
                   and row['qc_issue_count'] == 1 and row['qc_severity'] == 'HIGH'
                   for row in response.json())
        assert len(queries) <= 8, f'Sample list made {len(queries)} SELECT queries'
    finally:
        event.remove(engine, 'before_cursor_execute', record)
