import io
import json
import pytest

def test_bdd100k_upload_and_qc_flow(client):
    # 1. Prepare BDD100K test data
    bdd_sample = [
        {
            "name": "bdd_frame_01.jpg",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 100, "y1": 100, "x2": 250, "y2": 200}
                },
                {
                    "category": "bus",
                    "box2d": {"x1": 400, "y1": 150, "x2": 700, "y2": 380}
                }
            ]
        },
        {
            "name": "bdd_frame_02.jpg",
            "labels": [
                {
                    "category": "pedestrian",
                    "box2d": {"x1": 50, "y1": 200, "x2": 90, "y2": 310}
                }
            ]
        }
    ]
    bdd_json_bytes = json.dumps(bdd_sample).encode('utf-8')

    upload_res = client.post(
        '/api/datasets/upload',
        data={'name': 'BDD100K QC Test Dataset', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(bdd_json_bytes), 'application/json'))]
    )
    assert upload_res.status_code == 200
    dataset_data = upload_res.json()
    dataset_id = dataset_data['id']
    assert dataset_data['format'] == 'BDD100K'
    assert dataset_data['sample_count'] == 2

    # 2. Ingest Model Predictions (Simulating Detector Output)
    predictions_payload = {
        "predictions": [
            {
                "file_name": "bdd_frame_01.jpg",
                "predictions": [
                    # Agreement on car
                    {"label": "car", "confidence": 0.94, "geometry": {"x1": 102, "y1": 100, "x2": 248, "y2": 202}},
                    # Disagreement: human said 'bus', model predicts 'truck' with high confidence 0.89 -> WRONG_CLASS
                    {"label": "truck", "confidence": 0.89, "geometry": {"x1": 405, "y1": 152, "x2": 695, "y2": 378}},
                    # Missing object: model sees a car annotator forgot to label -> MISSING_OBJECT
                    {"label": "car", "confidence": 0.91, "geometry": {"x1": 800, "y1": 200, "x2": 950, "y2": 300}}
                ]
            },
            {
                "file_name": "bdd_frame_02.jpg",
                "predictions": [
                    # Agreement on pedestrian
                    {"label": "pedestrian", "confidence": 0.93, "geometry": {"x1": 52, "y1": 198, "x2": 88, "y2": 312}}
                ]
            }
        ]
    }

    pred_res = client.post(f'/api/datasets/{dataset_id}/qc/predictions', json=predictions_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data['predictions_count'] == 4

    # 3. Check QC Report
    audit_summary = pred_data['audit_summary']
    assert audit_summary['total_samples'] == 2
    assert audit_summary['flagged_samples_count'] == 1
    assert audit_summary['clean_samples_count'] == 1
    assert audit_summary['workload_reduction_percentage'] == 50.0
    assert audit_summary['total_missing_candidates'] == 1
    assert audit_summary['total_wrong_class_candidates'] == 1

    # 4. Check Prioritized Review Queue
    queue_res = client.get(f'/api/datasets/{dataset_id}/qc/queue')
    assert queue_res.status_code == 200
    queue = queue_res.json()
    assert len(queue) == 1
    top_sample = queue[0]
    assert top_sample['file_name'] == 'bdd_frame_01.jpg'
    assert top_sample['qc_severity'] == 'HIGH'
    assert len(top_sample['qc_issues']) == 2

    # Identify issues
    missing_issue = next(i for i in top_sample['qc_issues'] if i['issue_type'] == 'MISSING_OBJECT')
    wrong_issue = next(i for i in top_sample['qc_issues'] if i['issue_type'] == 'WRONG_CLASS')
    assert missing_issue['suggested_label'] == 'car'
    assert wrong_issue['human_label'] == 'bus'
    assert wrong_issue['suggested_label'] == 'truck'

    # 5. Reviewer resolves issues (One-click Accept)
    # Accept missing object -> adds new annotation
    res1 = client.put(f"/api/qc/issues/{missing_issue['id']}", json={'status': 'ACCEPTED', 'reviewer_note': 'Confirmed missed car in background'})
    assert res1.status_code == 200
    assert res1.json()['status'] == 'ACCEPTED'

    # Accept wrong class -> updates existing annotation from bus to truck
    res2 = client.put(f"/api/qc/issues/{wrong_issue['id']}", json={'status': 'ACCEPTED', 'reviewer_note': 'Re-classified as heavy truck'})
    assert res2.status_code == 200
    assert res2.json()['status'] == 'ACCEPTED'

    # Verify that the sample now has 3 annotations and the bus is now truck!
    sample_detail = client.get(f"/api/samples/{top_sample['id']}").json()
    labels = [a['label'] for a in sample_detail['annotations']]
    assert len(labels) == 3
    assert 'truck' in labels
    assert 'bus' not in labels

def test_synthetic_benchmark_endpoint(client):
    # Upload a dataset
    data = [
        {"name": "frame_a.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]},
        {"name": "frame_b.jpg", "labels": [{"category": "bus", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'Synth-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    synth_res = client.post(f'/api/datasets/{dataset_id}/qc/synthetic-benchmark?missing_ratio=0.5&wrong_class_ratio=0.5')
    assert synth_res.status_code == 200
    synth_data = synth_res.json()
    assert synth_data['predictions_generated'] > 0
    assert 'audit_summary' in synth_data

def test_detect_endpoint(client):
    data = [
        {"name": "frame_c.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'Detect-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']
    detect_res = client.post(f'/api/datasets/{dataset_id}/qc/detect')
    assert detect_res.status_code == 200
    assert 'predictions_generated' in detect_res.json()
    assert 'audit_summary' in detect_res.json()

