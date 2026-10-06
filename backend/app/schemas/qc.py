from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field

class BBoxGeometry(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class QCConfig(BaseModel):
    iou_match_threshold: float = Field(0.50, ge=0.1, le=0.9, description="IoU threshold to consider box referring to same object")
    iou_unmatched_threshold: float = Field(0.20, ge=0.0, le=0.5, description="Maximum IoU for prediction to be considered missing object")
    min_conf_missing: float = Field(0.40, ge=0.1, le=1.0, description="Minimum model confidence to flag missing object")
    min_conf_wrong_class: float = Field(0.45, ge=0.1, le=1.0, description="Minimum model confidence to flag wrong class")
    min_box_area: float = Field(100.0, ge=0.0, description="Minimum bounding box area in pixels to avoid extreme micro artifacts")
    taxonomy_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "person": "pedestrian",
            "pedestrian": "pedestrian",
            "car": "car",
            "automobile": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "motor": "motorcycle",
            "motorbike": "motorcycle",
            "bicycle": "bicycle",
            "bike": "bicycle",
            "rider": "rider",
            "cyclist": "rider",
            "traffic light": "traffic light",
            "traffic light green": "traffic light",
            "traffic light red": "traffic light",
            "traffic sign": "traffic sign",
            "stop sign": "traffic sign",
            "train": "train",
        }
    )
    class_reliability: Dict[str, float] = Field(
        default_factory=lambda: {
            "car": 1.0,
            "pedestrian": 0.95,
            "bus": 0.90,
            "truck": 0.90,
            "motorcycle": 0.85,
            "bicycle": 0.85,
            "rider": 0.80,
            "traffic sign": 0.80,
            "traffic light": 0.75,
            "train": 0.85,
        }
    )

class PredictionItem(BaseModel):
    label: str
    geometry: BBoxGeometry
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_model: Optional[str] = "pretrained"

class SamplePredictionsBatch(BaseModel):
    file_name: str
    predictions: List[PredictionItem]

class IngestPredictionsRequest(BaseModel):
    predictions: List[SamplePredictionsBatch]
    source_model: Optional[str] = "yolov8-bdd100k"

class QCIssueResponse(BaseModel):
    id: int
    sample_id: int
    issue_type: Literal['MISSING_OBJECT', 'WRONG_CLASS']
    location: Dict[str, float]
    human_label: Optional[str] = None
    suggested_label: str
    annotation_id: Optional[int] = None
    prediction_id: Optional[int] = None
    qc_score: float
    evidence: Dict[str, Any]
    status: Literal['PENDING', 'ACCEPTED', 'REJECTED']
    reviewer_note: str

class QCIssueResolveRequest(BaseModel):
    status: Literal['ACCEPTED', 'REJECTED']
    reviewer_note: Optional[str] = ""

class QCSampleSummary(BaseModel):
    sample_id: int
    file_name: str
    qc_score: float
    severity: Literal['HIGH', 'MEDIUM', 'LOW', 'CLEAN']
    missing_count: int
    wrong_class_count: int
    total_issues: int
    issues: List[QCIssueResponse]

class QCAuditReport(BaseModel):
    dataset_id: int
    total_samples: int
    flagged_samples_count: int
    clean_samples_count: int
    workload_reduction_percentage: float
    total_missing_candidates: int
    total_wrong_class_candidates: int
    total_issues: int
    severity_breakdown: Dict[str, int]
    top_suspicious_samples: List[QCSampleSummary]
