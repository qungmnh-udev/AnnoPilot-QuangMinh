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

def test_cvat_config_and_deep_link(client):
    data = [
        {"name": "frame_000042.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'CVAT-Link-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    # Update CVAT configuration
    cvat_conf = client.put(
        f'/api/datasets/{dataset_id}/cvat-config',
        json={
            'cvat_base_url': 'http://localhost:8080',
            'cvat_task_id': 101,
            'cvat_job_id': 202
        }
    )
    assert cvat_conf.status_code == 200
    dataset_info = cvat_conf.json()
    assert dataset_info['cvat_task_id'] == 101
    assert dataset_info['cvat_job_id'] == 202
    assert dataset_info['cvat_base_url'] == 'http://localhost:8080'

    # Run synthetic benchmark to inject QC issues
    synth_res = client.post(f'/api/datasets/{dataset_id}/qc/synthetic-benchmark?missing_ratio=1.0&wrong_class_ratio=0.0')
    assert synth_res.status_code == 200

    # Check queue issues have cvat_url pointing to frame 42
    queue = client.get(f'/api/datasets/{dataset_id}/qc/queue').json()
    assert len(queue) == 1
    sample = queue[0]
    assert sample['frame_number'] == 42
    assert sample['cvat_url'] == 'http://localhost:8080/tasks/101/jobs/202?frame=42'

    assert len(sample['qc_issues']) > 0
    issue = sample['qc_issues'][0]
    assert issue['frame_number'] == 42
    assert issue['cvat_url'] == 'http://localhost:8080/tasks/101/jobs/202?frame=42'

    # Test resolving with RESOLVED status
    res_put = client.put(
        f"/api/qc/issues/{issue['id']}",
        json={'status': 'RESOLVED', 'reviewer_note': 'Fixed directly in CVAT'}
    )
    assert res_put.status_code == 200
    assert res_put.json()['status'] == 'RESOLVED'

    # Also test get_cvat_link endpoint
    link_res = client.get(f"/api/qc/issues/{issue['id']}/cvat-link")
    assert link_res.status_code == 200
    assert link_res.json()['cvat_url'] == 'http://localhost:8080/tasks/101/jobs/202?frame=42'

def test_single_frame_audit_and_cvat_verify(client):
    # Upload 2 frames
    data = [
        {"name": "frame_01.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]},
        {"name": "frame_02.jpg", "labels": [{"category": "bus", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'Single-Frame-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']
    samples_res = client.get(f'/api/datasets/{dataset_id}/samples')
    samples = samples_res.json()
    assert len(samples) == 2
    s1 = samples[0]

    # Ingest a prediction on s1 that flags WRONG_CLASS
    pred_payload = {
        "predictions": [
            {
                "file_name": "frame_01.jpg",
                "predictions": [
                    {"label": "truck", "confidence": 0.91, "geometry": {"x1": 52, "y1": 50, "x2": 148, "y2": 150}}
                ]
            }
        ]
    }
    client.post(f'/api/datasets/{dataset_id}/qc/predictions', json=pred_payload)

    # Call on-demand single frame audit endpoint
    audit_res = client.post(f'/api/datasets/{dataset_id}/qc/frame/{s1["id"]}/audit?force_repredict=false')
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert 'sample_summary' in audit_data
    summary = audit_data['sample_summary']
    assert summary['sample_id'] == s1['id']
    assert summary['wrong_class_count'] == 1
    assert summary['total_issues'] == 1

    # Call CVAT verify endpoint on this frame
    verify_res = client.post(f'/api/datasets/{dataset_id}/qc/frame/{s1["id"]}/cvat-verify')
    assert verify_res.status_code == 200
    verify_data = verify_res.json()
    assert verify_data['sample_id'] == s1['id']
    assert verify_data['is_verified'] is False
    assert verify_data['remaining_issues_count'] == 1

def test_incremental_queue_lifecycle(client):
    data = [
        {"name": "stream_frame_1.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]},
        {"name": "stream_frame_2.jpg", "labels": [{"category": "bus", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'Queue-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    # Check status initially IDLE
    status_res = client.get(f'/api/datasets/{dataset_id}/qc/queue/status')
    assert status_res.status_code == 200
    assert status_res.json()['status'] in ('IDLE', 'COMPLETED')

    # Start queue
    start_res = client.post(f'/api/datasets/{dataset_id}/qc/queue/start', json={'conf_thresh': 0.40, 'force_repredict': False})
    assert start_res.status_code == 200
    status = start_res.json()
    assert status['status'] in ('RUNNING', 'COMPLETED')
    assert status['total_frames'] == 2

    # Stop queue
    stop_res = client.post(f'/api/datasets/{dataset_id}/qc/queue/stop')
    assert stop_res.status_code == 200
    assert stop_res.json()['status'] in ('PAUSED', 'COMPLETED')

def test_cvat_sync_auto_resolution(client):
    # 1. Dataset with an initial annotation 'bus'
    initial_data = [
        {"name": "frame_sync_01.jpg", "labels": [{"category": "bus", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'CVAT-Sync-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(initial_data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    # 2. Detector predicts 'truck' with high confidence -> WRONG_CLASS issue created
    pred_payload = {
        "predictions": [
            {
                "file_name": "frame_sync_01.jpg",
                "predictions": [
                    {"label": "truck", "confidence": 0.92, "geometry": {"x1": 102, "y1": 100, "x2": 298, "y2": 250}}
                ]
            }
        ]
    }
    client.post(f'/api/datasets/{dataset_id}/qc/predictions', json=pred_payload)

    # Verify 1 PENDING issue
    queue = client.get(f'/api/datasets/{dataset_id}/qc/queue').json()
    assert len(queue) == 1
    assert queue[0]['qc_issues'][0]['status'] == 'PENDING'

    # 3. User corrected on CVAT: label changed from 'bus' to 'truck'
    updated_cvat_data = [
        {"name": "frame_sync_01.jpg", "labels": [{"category": "truck", "box2d": {"x1": 100, "y1": 100, "x2": 300, "y2": 250}}]}
    ]
    updated_bytes = json.dumps(updated_cvat_data).encode('utf-8')

    sync_res = client.post(
        f'/api/datasets/{dataset_id}/qc/cvat-sync',
        files=[('file', ('cvat_updated.json', io.BytesIO(updated_bytes), 'application/json'))]
    )
    assert sync_res.status_code == 200
    sync_result = sync_res.json()
    assert sync_result['status'] == 'success'
    assert sync_result['auto_resolved_issues_count'] >= 1
    assert sync_result['remaining_issues_count'] == 0

def test_cvat_push_endpoint(client, monkeypatch):
    import urllib.request
    from unittest.mock import MagicMock

    data = [
        {"name": "push_frame_01.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'CVAT-Push-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    # Mock urllib.request.urlopen for CVAT API calls
    mock_urlopen = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.side_effect = [
        json.dumps({'results': [{'id': 202}]}).encode(), # /api/jobs
        json.dumps({'results': [{'id': 13, 'name': 'vehicle'}]}).encode(), # /api/labels
        json.dumps({'tracks': [], 'tags': [], 'version': 1}).encode(), # get existing annotations
        json.dumps({'status': 'ok'}).encode(), # put annotations
    ]
    mock_urlopen.return_value.__enter__.return_value = mock_response
    monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)

    push_res = client.post(
        f'/api/datasets/{dataset_id}/qc/cvat-push',
        json={
            'token': 'TEST_PAT_TOKEN',
            'cvat_base_url': 'http://localhost:8080',
            'cvat_task_id': 101,
            'mode': 'replace'
        }
    )
    assert push_res.status_code == 200
    push_data = push_res.json()
    assert push_data['status'] == 'success'
    assert push_data['pushed_shapes_count'] == 1
    assert push_data['cvat_task_id'] == 101

def test_qc_report_caching_and_speed(client):
    # Upload sample dataset
    data = [
        {"name": "speed_frame_1.jpg", "labels": [{"category": "car", "box2d": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}]}
    ]
    res = client.post(
        '/api/datasets/upload',
        data={'name': 'Speed-Test', 'format': 'BDD100K', 'task_type': 'BBOX_2D'},
        files=[('annotations', ('labels.json', io.BytesIO(json.dumps(data).encode()), 'application/json'))]
    )
    dataset_id = res.json()['id']

    # Ingest prediction
    pred_res = client.post(f'/api/datasets/{dataset_id}/qc/predictions', json={
        "predictions": [
            {
                "file_name": "speed_frame_1.jpg",
                "predictions": [
                    {"label": "truck", "confidence": 0.90, "geometry": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}}
                ]
            }
        ]
    })
    assert pred_res.status_code == 200

    # First call: builds report & caches
    rep1 = client.get(f'/api/datasets/{dataset_id}/qc/report')
    assert rep1.status_code == 200
    data1 = rep1.json()
    assert data1['total_samples'] == 1
    assert data1['flagged_samples_count'] == 1
    assert data1['total_issues'] == 1

    # Second call: served from in-memory cache
    rep2 = client.get(f'/api/datasets/{dataset_id}/qc/report')
    assert rep2.status_code == 200
    assert rep2.json() == data1

    # Resolve issue: should invalidate cache
    issue_id = data1['top_suspicious_samples'][0]['issues'][0]['id']
    resolve_res = client.put(f'/api/qc/issues/{issue_id}', json={'status': 'ACCEPTED', 'reviewer_note': 'Confirmed truck'})
    assert resolve_res.status_code == 200

    # Next report call: should reflect ACCEPTED status
    rep3 = client.get(f'/api/datasets/{dataset_id}/qc/report')
    assert rep3.status_code == 200
    data3 = rep3.json()
    assert data3['top_suspicious_samples'][0]['issues'][0]['status'] == 'ACCEPTED'




