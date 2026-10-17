import pytest
from app.services.ensemble import (
    DetectionBox,
    FusedBox,
    weighted_box_fusion,
    fuse_multi_model_predictions,
    get_dataset_prediction_models,
    run_ensemble_from_existing_predictions,
)
from app.models import Dataset, Sample, Prediction


def test_wbf_tier_1_high_consensus():
    """Two models detect the same car with high overlap and confidence -> Tier 1."""
    box_yolo = DetectionBox(label="car", confidence=0.92, x1=10.0, y1=10.0, x2=50.0, y2=50.0, source_model="yolo26n")
    box_rtdetr = DetectionBox(label="car", confidence=0.88, x1=12.0, y1=11.0, x2=49.0, y2=51.0, source_model="rt-detr")

    fused = weighted_box_fusion(
        {"yolo26n": [box_yolo], "rt-detr": [box_rtdetr]},
        iou_threshold=0.55,
        confidence_threshold=0.30,
    )

    assert len(fused) == 1
    fb = fused[0]
    assert fb.label == "car"
    assert fb.tier == 1
    assert fb.num_models_agreed == 2
    assert set(fb.contributing_models) == {"yolo26n", "rt-detr"}
    assert 10.0 < fb.x1 < 12.0
    assert 10.0 < fb.y1 < 11.0


def test_wbf_tier_2_class_disagreement():
    """Two models detect an object at same spot but disagree on class -> Tier 2."""
    box_yolo = DetectionBox(label="car", confidence=0.90, x1=10.0, y1=10.0, x2=50.0, y2=50.0, source_model="yolo26n")
    box_rtdetr = DetectionBox(label="truck", confidence=0.85, x1=12.0, y1=11.0, x2=49.0, y2=52.0, source_model="rt-detr")

    fused = weighted_box_fusion(
        {"yolo26n": [box_yolo], "rt-detr": [box_rtdetr]},
        iou_threshold=0.55,
    )

    assert len(fused) == 1
    fb = fused[0]
    assert fb.tier == 2
    assert "Class disagreement" in fb.tier_reason
    assert "car" in fb.model_votes
    assert "truck" in fb.model_votes


def test_wbf_tier_3_single_model():
    """Only one model detects an object -> Tier 3."""
    box_yolo = DetectionBox(label="traffic sign", confidence=0.65, x1=100.0, y1=100.0, x2=130.0, y2=130.0, source_model="yolo26n")

    fused = weighted_box_fusion(
        {"yolo26n": [box_yolo], "rt-detr": []},
        iou_threshold=0.55,
    )

    assert len(fused) == 1
    fb = fused[0]
    assert fb.tier == 3
    assert fb.contributing_models == ["yolo26n"]
    assert "Only detected by yolo26n" in fb.tier_reason


def test_wbf_weights_influence():
    """Heavier weighted model has more influence on fused coordinates."""
    box_a = DetectionBox(label="car", confidence=0.90, x1=0.0, y1=0.0, x2=10.0, y2=10.0, source_model="model_a")
    box_b = DetectionBox(label="car", confidence=0.90, x1=10.0, y1=10.0, x2=20.0, y2=20.0, source_model="model_b")

    # Give model_a weight 3.0, model_b weight 1.0
    fused = weighted_box_fusion(
        {"model_a": [box_a], "model_b": [box_b]},
        model_weights={"model_a": 3.0, "model_b": 1.0},
        iou_threshold=0.0, # force clustering
    )

    assert len(fused) == 1
    # x1 should be heavily biased towards model_a's 0.0: (0 * 3*0.9 + 10 * 1*0.9) / (3*0.9 + 1*0.9) = 10 / 4 = 2.5
    assert abs(fused[0].x1 - 2.5) < 0.1


def test_api_ensemble_endpoints(client, db_session):
    """Test get ensemble models and run ensemble endpoints."""
    # Create dataset with sample and two model predictions
    ds = Dataset(name="Ensemble Test Dataset", format="coco", task_type="detection")
    db_session.add(ds)
    db_session.commit()

    sample = Sample(dataset_id=ds.id, file_name="frame_001.jpg", width=1280, height=720, task_type="detection")
    db_session.add(sample)
    db_session.commit()

    # Model 1 prediction
    p1 = Prediction(
        sample_id=sample.id,
        label="car",
        confidence=0.90,
        geometry={"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
        source_model="yolo26n"
    )
    # Model 2 prediction
    p2 = Prediction(
        sample_id=sample.id,
        label="car",
        confidence=0.88,
        geometry={"x1": 12.0, "y1": 11.0, "x2": 49.0, "y2": 51.0},
        source_model="rt-detr"
    )
    db_session.add_all([p1, p2])
    db_session.commit()

    # 1. Test GET /api/datasets/{id}/ensemble/models
    res = client.get(f"/api/datasets/{ds.id}/ensemble/models")
    assert res.status_code == 200
    data = res.json()
    assert set(data["models"]) == {"rt-detr", "yolo26n"}
    assert data["counts"]["yolo26n"] == 1
    assert data["counts"]["rt-detr"] == 1

    # 2. Test POST /api/datasets/{id}/ensemble/run
    run_res = client.post(
        f"/api/datasets/{ds.id}/ensemble/run",
        json={
            "iou_threshold": 0.55,
            "confidence_threshold": 0.30,
            "ensemble_source_model": "ensemble_wbf_test",
            "run_audit": False,
        }
    )
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["status"] == "success"
    assert run_data["total_fused_predictions"] == 1
    assert run_data["tier_breakdown"]["tier_1_high_consensus"] == 1


def test_qc_audit_prioritizes_ensemble_and_avoids_multi_model_clutter(client, db_session):
    """Verify that multi-model predictions auto-fuse and generate only 1 consolidated issue instead of duplicate issues."""
    ds = Dataset(name="Consensus QC Dataset", format="bdd100k", task_type="detection")
    db_session.add(ds)
    db_session.commit()

    sample = Sample(dataset_id=ds.id, file_name="multi_001.jpg", width=1280, height=720, task_type="detection")
    db_session.add(sample)
    db_session.commit()

    # Two models detect an unannotated pedestrian at almost same spot
    p1 = Prediction(
        sample_id=sample.id,
        label="pedestrian",
        confidence=0.89,
        geometry={"x1": 50.0, "y1": 50.0, "x2": 80.0, "y2": 150.0},
        source_model="yolo26n"
    )
    p2 = Prediction(
        sample_id=sample.id,
        label="pedestrian",
        confidence=0.85,
        geometry={"x1": 52.0, "y1": 49.0, "x2": 79.0, "y2": 151.0},
        source_model="fasterrcnn_mobilenet_v3"
    )
    db_session.add_all([p1, p2])
    db_session.commit()

    # Trigger single frame audit - it should auto-fuse via WBF
    res = client.post(f"/api/datasets/{ds.id}/qc/frame/{sample.id}/audit")
    assert res.status_code == 200

    # Fetch sample details
    sample_res = client.get(f"/api/samples/{sample.id}")
    assert sample_res.status_code == 200
    s_data = sample_res.json()

    # Exactly 1 consolidated issue should exist, NOT 2 separate issues!
    issues = s_data.get("qc_issues", [])
    assert len(issues) == 1, f"Expected 1 consolidated issue from ensemble, got {len(issues)}"
    assert issues[0]["issue_type"] == "MISSING_OBJECT"
    assert issues[0]["suggested_label"] == "pedestrian"
    evidence = issues[0]["evidence"]
    assert evidence.get("consensus_tier") == 1
    assert evidence.get("num_models_agreed") == 2

    # Test Batch 1-Click Resolve on this frame
    batch_res = client.post(
        f"/api/samples/{sample.id}/qc/batch-resolve",
        json={"consensus_tier": 1, "status": "ACCEPTED"}
    )
    assert batch_res.status_code == 200
    assert batch_res.json()["resolved_count"] == 1

    # Verify annotation was automatically created from 1-Click resolve
    refreshed = client.get(f"/api/samples/{sample.id}").json()
    assert len(refreshed["annotations"]) == 1
    assert refreshed["annotations"][0]["label"] == "pedestrian"
    assert refreshed["qc_issues"][0]["status"] == "ACCEPTED"

