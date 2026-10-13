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
            "vehicle": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "motor": "motorcycle",
            "motorbike": "motorcycle",
            "motorcyclist": "rider",
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
            "vehicle": 1.0,
            "pedestrian": 0.95,
            "bus": 0.90,
            "truck": 0.90,
            "motorcycle": 0.85,
            "motorcyclist": 0.85,
            "bicycle": 0.85,
            "rider": 0.80,
            "traffic sign": 0.80,
            "traffic light": 0.75,
            "train": 0.85,
        }
    )

class CVATConfigRequest(BaseModel):
    cvat_base_url: Optional[str] = "http://localhost:8080"
    cvat_task_id: Optional[int] = None
    cvat_job_id: Optional[int] = None
    cvat_token: Optional[str] = "ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO"

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
    frame_number: Optional[int] = None
    cvat_url: Optional[str] = None
    status: Literal['PENDING', 'ACCEPTED', 'REJECTED', 'RESOLVED', 'FALSE_POSITIVE']
    reviewer_note: str

class QCIssueResolveRequest(BaseModel):
    status: Literal['ACCEPTED', 'REJECTED', 'RESOLVED', 'FALSE_POSITIVE', 'PENDING']
    reviewer_note: Optional[str] = ""
    sample_id: Optional[int] = None
    issue_type: Optional[str] = None
    suggested_label: Optional[str] = None

class QCBatchResolveRequest(BaseModel):
    sample_id: Optional[int] = None
    dataset_id: Optional[int] = None
    issue_ids: Optional[List[int]] = None
    consensus_tier: Optional[int] = None  # e.g. 1 for Tier 1 high consensus
    status: Literal['ACCEPTED', 'REJECTED', 'RESOLVED', 'FALSE_POSITIVE'] = 'ACCEPTED'
    reviewer_note: Optional[str] = "Batch resolved via multi-model consensus"

class QCSampleSummary(BaseModel):
    sample_id: int
    file_name: str
    frame_number: Optional[int] = None
    cvat_url: Optional[str] = None
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

# --- Incremental Frame Processing & CVAT Sync Schemas (P0) ---

class StartQueueRequest(BaseModel):
    conf_thresh: Optional[float] = 0.35
    sample_ids: Optional[List[int]] = None
    force_repredict: Optional[bool] = False

class FrameQueueStatusResponse(BaseModel):
    dataset_id: int
    status: Literal['IDLE', 'RUNNING', 'PAUSED', 'COMPLETED', 'ERROR']
    total_frames: int
    processed_frames: int
    progress_percent: float
    current_frame_id: Optional[int] = None
    current_frame_name: Optional[str] = None
    flagged_frames_count: int = 0
    recent_audited_frames: List[QCSampleSummary] = Field(default_factory=list)
    error_message: Optional[str] = None

class FrameAuditResponse(BaseModel):
    sample_summary: QCSampleSummary
    predictions_count: int

class CVATSyncResponse(BaseModel):
    status: str
    message: str
    dataset_id: int
    updated_frames_count: int
    auto_resolved_issues_count: int
    remaining_issues_count: int
    synced_at: str

class CVATVerifyFrameResponse(BaseModel):
    sample_id: int
    frame_number: Optional[int] = None
    is_verified: bool
    resolved_issues_count: int
    remaining_issues_count: int
    message: str
    issues: List[QCIssueResponse]

class CVATPushRequest(BaseModel):
    token: Optional[str] = None
    cvat_base_url: Optional[str] = None
    cvat_task_id: Optional[int] = None
    cvat_job_id: Optional[int] = None
    mode: Optional[Literal['create', 'replace']] = 'create'
    target_source: Optional[Literal['merged', 'predictions', 'annotations']] = 'merged'

class CVATPushResponse(BaseModel):
    status: str
    message: str
    cvat_task_id: Optional[int] = None
    cvat_job_id: Optional[int] = None
    pushed_shapes_count: int
    cvat_url: Optional[str] = None
    target_source: Optional[str] = 'merged'
    mode: Optional[str] = 'create'

# --- Multi-Model Ensemble & WBF Pre-labeling Schemas ---

class EnsembleRunRequest(BaseModel):
    model_names: Optional[List[str]] = None
    model_weights: Optional[Dict[str, float]] = None
    iou_threshold: float = Field(0.55, ge=0.1, le=0.95, description="IoU threshold for clustering boxes across models")
    confidence_threshold: float = Field(0.30, ge=0.0, le=1.0, description="Minimum fused confidence to keep")
    high_consensus_conf: float = Field(0.85, ge=0.1, le=1.0, description="Confidence threshold for Tier 1 high consensus")
    high_consensus_iou: float = Field(0.60, ge=0.1, le=1.0, description="IoU threshold for Tier 1 high consensus")
    ensemble_source_model: str = Field("ensemble_wbf_v1", description="Identifier for fused predictions")
    run_audit: bool = Field(True, description="Automatically re-run QC audit after fusion")

class DatasetModelsResponse(BaseModel):
    dataset_id: int
    models: List[str]
    counts: Dict[str, int]
    has_ensemble: bool = False

class EnsembleResponse(BaseModel):
    status: str
    message: Optional[str] = None
    ensemble_source_model: Optional[str] = None
    models_used: Optional[List[str]] = None
    total_fused_predictions: Optional[int] = 0
    tier_breakdown: Optional[Dict[str, int]] = None
    files_processed: Optional[int] = 0
    audit_summary: Optional[Dict[str, Any]] = None

class TriadModelInfo(BaseModel):
    id: str
    label: str
    family: str
    architecture: str
    default_weight: float
    description: str

class TriadRunRequest(BaseModel):
    models: Optional[List[str]] = ["yolov8n", "fasterrcnn_mobilenet_v3", "rtdetr_l"]
    weights: Optional[Dict[str, float]] = {
        "yolov8n": 1.0,
        "fasterrcnn_mobilenet_v3": 1.0,
        "rtdetr_l": 1.2
    }
    conf_thresh: float = Field(0.35, ge=0.05, le=1.0)
    iou_thresh: float = Field(0.55, ge=0.1, le=0.95)
    sample_limit: Optional[int] = Field(None, description="Limit number of frames for fast testing (None for all)")
    auto_ensemble: bool = Field(True, description="Automatically run Tri-Tier WBF ensemble after inference")

class TriadRunResponse(BaseModel):
    success: bool
    frames_processed: int
    selected_models: List[str]
    model_counts: Dict[str, int]
    total_predictions: int
    ensemble_result: Optional[Dict[str, Any]] = None
    triad_specs: Optional[Dict[str, Any]] = None



