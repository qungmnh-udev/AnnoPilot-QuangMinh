import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.triad_detector import COCO_TO_BDD100K, BDD100K_CLASSES, TRIAD_SPECS
from app.models import Dataset, Sample, Prediction
from app.database import SessionLocal

client = TestClient(app)


def test_coco_to_bdd100k_taxonomy_mapping():
    """Verify all mapped COCO classes are valid BDD100K driving classes."""
    for coco_cls, bdd_cls in COCO_TO_BDD100K.items():
        assert bdd_cls in BDD100K_CLASSES, f"Class {bdd_cls} mapped from {coco_cls} is not in BDD100K taxonomy"


def test_triad_architectural_diversity_specs():
    """Verify Option B Triad consists of 3 distinct architecture families."""
    assert "yolo26n" in TRIAD_SPECS
    assert "fasterrcnn_mobilenet_v3" in TRIAD_SPECS
    assert "rtdetr_l" in TRIAD_SPECS

    families = {spec["family"] for spec in TRIAD_SPECS.values()}
    assert "CNN 1-Stage" in families
    assert "CNN 2-Stage" in families
    assert "Vision Transformer" in families


def test_api_triad_specs_endpoint():
    """Verify GET /api/ensemble/triad/specs returns all 3 models."""
    response = client.get("/api/ensemble/triad/specs")
    assert response.status_code == 200
    data = response.json()
    assert "yolo26n" in data
    assert "fasterrcnn_mobilenet_v3" in data
    assert "rtdetr_l" in data
    assert data["rtdetr_l"]["family"] == "Vision Transformer"


def test_api_triad_run_endpoint_validation():
    """Verify POST /api/datasets/{id}/ensemble/run-triad handles non-existent dataset with 404."""
    response = client.post("/api/datasets/999999/ensemble/run-triad", json={})
    assert response.status_code == 404
