import pytest
from app.services.qc_engine import (
    calculate_box_iou,
    calculate_box_area,
    normalize_label,
    match_boxes,
    detect_sample_qc_issues,
    compute_sample_qc_score,
    run_dataset_qc_audit
)
from app.schemas.qc import QCConfig
from app.models import Dataset, Sample, Annotation, Prediction, QCIssue

def test_box_iou_and_area():
    b1 = {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}
    b2 = {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}
    assert calculate_box_area(b1) == 100
    assert calculate_box_iou(b1, b2) == 1.0

    b3 = {'x1': 10, 'y1': 10, 'x2': 20, 'y2': 20}
    assert calculate_box_iou(b1, b3) == 0.0

    b4 = {'x1': 5, 'y1': 0, 'x2': 15, 'y2': 10}
    # intersection: 5*10 = 50. area1=100, area4=100. union=150. iou = 50/150 = 1/3
    assert abs(calculate_box_iou(b1, b4) - (1/3)) < 1e-4

def test_normalize_label():
    mapping = {'person': 'pedestrian', 'automobile': 'car'}
    assert normalize_label(' Person ', mapping) == 'pedestrian'
    assert normalize_label('AUTOMOBILE', mapping) == 'car'
    assert normalize_label('truck', mapping) == 'truck'

def test_detect_wrong_class_candidate():
    cfg = QCConfig()
    ann = Annotation(
        id=1,
        sample_id=1,
        label='car',
        shape_type='BBOX_2D',
        geometry={'x1': 100, 'y1': 100, 'x2': 300, 'y2': 300}
    )
    # Model predicts same location but as 'truck' with high confidence 0.90
    pred = Prediction(
        id=1,
        sample_id=1,
        label='truck',
        geometry={'x1': 105, 'y1': 100, 'x2': 305, 'y2': 300},
        confidence=0.90,
        source_model='yolo'
    )

    issues = detect_sample_qc_issues(1, [ann], [pred], cfg)
    assert len(issues) == 1
    issue = issues[0]
    assert issue['issue_type'] == 'WRONG_CLASS'
    assert issue['human_label'] == 'car'
    assert issue['suggested_label'] == 'truck'
    assert issue['qc_score'] > 0.70
    assert issue['evidence']['iou'] > 0.85

def test_detect_missing_object_candidate():
    cfg = QCConfig()
    ann = Annotation(
        id=1,
        sample_id=1,
        label='traffic sign',
        shape_type='BBOX_2D',
        geometry={'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}
    )
    # Model detects a prominent pedestrian on the opposite side
    pred = Prediction(
        id=1,
        sample_id=1,
        label='pedestrian',
        geometry={'x1': 400, 'y1': 200, 'x2': 460, 'y2': 350},
        confidence=0.88,
        source_model='yolo'
    )

    issues = detect_sample_qc_issues(1, [ann], [pred], cfg)
    assert len(issues) == 1
    issue = issues[0]
    assert issue['issue_type'] == 'MISSING_OBJECT'
    assert issue['human_label'] is None
    assert issue['suggested_label'] == 'pedestrian'
    assert issue['qc_score'] > 0.65
    assert issue['evidence']['max_iou_with_any_annotation'] == 0.0

def test_clean_sample_when_model_and_gt_agree():
    cfg = QCConfig()
    ann = Annotation(
        id=1,
        sample_id=1,
        label='car',
        shape_type='BBOX_2D',
        geometry={'x1': 100, 'y1': 100, 'x2': 250, 'y2': 250}
    )
    pred = Prediction(
        id=1,
        sample_id=1,
        label='car',
        geometry={'x1': 102, 'y1': 98, 'x2': 248, 'y2': 252},
        confidence=0.92,
        source_model='yolo'
    )

    issues = detect_sample_qc_issues(1, [ann], [pred], cfg)
    assert len(issues) == 0

    score, severity = compute_sample_qc_score(issues)
    assert score == 0.0
    assert severity == 'CLEAN'

def test_qc_audit_dataset(db_session):
    # Setup dataset in test db
    dataset = Dataset(name='BDD100K-QC-Test', format='BDD100K', task_type='BBOX_2D')
    db_session.add(dataset)
    db_session.flush()

    # Sample 1: Has a wrong class error
    s1 = Sample(dataset_id=dataset.id, file_name='sample_1.jpg', width=1280, height=720, task_type='BBOX_2D')
    # Sample 2: Clean sample
    s2 = Sample(dataset_id=dataset.id, file_name='sample_2.jpg', width=1280, height=720, task_type='BBOX_2D')
    db_session.add_all([s1, s2])
    db_session.flush()

    # Annotations
    a1 = Annotation(sample_id=s1.id, label='bus', shape_type='BBOX_2D', geometry={'x1': 100, 'y1': 100, 'x2': 400, 'y2': 300})
    a2 = Annotation(sample_id=s2.id, label='car', shape_type='BBOX_2D', geometry={'x1': 200, 'y1': 200, 'x2': 350, 'y2': 300})
    db_session.add_all([a1, a2])

    # Predictions
    p1 = Prediction(sample_id=s1.id, label='truck', geometry={'x1': 102, 'y1': 99, 'x2': 398, 'y2': 301}, confidence=0.91, source_model='yolo')
    p2 = Prediction(sample_id=s2.id, label='car', geometry={'x1': 201, 'y1': 200, 'x2': 349, 'y2': 300}, confidence=0.95, source_model='yolo')
    db_session.add_all([p1, p2])
    db_session.commit()

    # Run audit
    report = run_dataset_qc_audit(db_session, dataset.id)
    assert report['total_samples'] == 2
    assert report['flagged_samples_count'] == 1
    assert report['clean_samples_count'] == 1
    assert report['workload_reduction_percentage'] == 50.0
    assert report['total_wrong_class_candidates'] == 1
    assert report['total_missing_candidates'] == 0

    # Verify ranked queue order
    top_sample = report['top_suspicious_samples'][0]
    assert top_sample['sample_id'] == s1.id
    assert top_sample['severity'] == 'HIGH'
    assert top_sample['wrong_class_count'] == 1
