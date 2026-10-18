export type Feature = {
  value: number | null;
  raw: unknown;
  available: boolean;
  reason_if_unavailable: string | null;
};
export type YOLOVariant = 'yolo26n' | 'yolo26s';
export type Analysis = {
  visual: {
    features: Record<string, Feature>;
    score: number | null;
    contributions: Record<string, number>;
  };
  per_type: Record<
    string,
    {
      features: Record<string, Feature>;
      score: number | null;
      contributions: Record<string, number>;
    }
  >;
  feature_contributions: Record<string, number>;
  rare_classes: string[];
};
export type Annotation = {
  id: number;
  label: string;
  shape_type: string;
  geometry: Record<string, any>;
  attributes: Record<string, unknown>;
  occluded: boolean | null;
  metadata: Record<string, unknown>;
};
export type Prediction = {
  id: number;
  label: string;
  geometry: { x1: number; y1: number; x2: number; y2: number };
  confidence: number;
  source_model: string;
};

export type QCIssue = {
  id: number;
  sample_id: number;
  issue_type: 'MISSING_OBJECT' | 'WRONG_CLASS';
  location: { x1: number; y1: number; x2: number; y2: number };
  human_label: string | null;
  suggested_label: string;
  annotation_id: number | null;
  prediction_id: number | null;
  qc_score: number;
  evidence: {
    iou?: number;
    max_iou_with_any_annotation?: number;
    model_confidence: number;
    reason: string;
    consensus_tier?: number;
    num_models_agreed?: number;
    tier_reason?: string;
    contributing_models?: string[];
    model_votes?: Record<string, number>;
    source_model?: string;
    [key: string]: any;
  };
  frame_number?: number | null;
  cvat_url?: string | null;
  status: 'PENDING' | 'ACCEPTED' | 'REJECTED' | 'RESOLVED' | 'FALSE_POSITIVE';
  reviewer_note: string;
};

export type Sample = {
  id: number;
  file_name: string;
  task_type: string;
  width: number | null;
  height: number | null;
  annotation_count: number;
  prediction_count?: number;
  qc_processing_state?: 'NOT_RUN' | 'PROCESSED' | 'ERROR';
  inference_state?: Record<string, { status: string; checkpoint?: string; error?: string }>;
  qc_score?: number;
  qc_severity?: 'HIGH' | 'MEDIUM' | 'LOW' | 'CLEAN';
  qc_issue_count?: number;
  qc_issue_counts?: { missing: number; wrong: number; resolved: number; total: number };
  frame_number?: number | null;
  cvat_url?: string | null;
  media_url: string | null;
  media_kind: string | null;
  annotation_difficulty: number | null;
  visual_difficulty: number | null;
  overall_difficulty: number | null;
  level: string;
  rare_classes: string[];
  analysis?: Analysis;
  annotations?: Annotation[];
  predictions?: Prediction[];
  qc_issues?: QCIssue[];
  sampling_reason?: string;
  assignment_id?: number;
  reviewer_id?: number;
  reviewer?: string;
  status?: string;
  note?: string;
  needs_regeneration?: boolean;
};

export type QCAuditReport = {
  dataset_id: number;
  total_samples: number;
  processed_samples_count?: number;
  inference_error_samples_count?: number;
  unprocessed_samples_count?: number;
  flagged_samples_count: number;
  clean_samples_count: number;
  workload_reduction_percentage: number;
  total_missing_candidates: number;
  total_wrong_class_candidates: number;
  total_issues: number;
  severity_breakdown: Record<string, number>;
  top_suspicious_samples: Array<{
    sample_id: number;
    file_name: string;
    qc_score: number;
    severity: 'HIGH' | 'MEDIUM' | 'LOW' | 'CLEAN';
    missing_count: number;
    wrong_class_count: number;
    total_issues: number;
    issues: QCIssue[];
  }>;
};

export type Dataset = {
  id: number;
  name: string;
  format: string;
  task_type: string;
  revision: number;
  sample_count: number;
  annotation_count: number;
  average_difficulty: number | null;
  levels: Record<string, number>;
  classes: Record<string, number>;
  annotation_types: Record<string, number>;
  rare_classes: string[];
  missing_media: number;
  warnings: string[];
  cvat_task_id?: number | null;
  cvat_job_id?: number | null;
  cvat_base_url?: string | null;
  cvat_token?: string | null;
};
export type Run = {
  id: number;
  config: unknown;
  needs_regeneration: boolean;
  selections: Sample[];
};
export type Reviewer = { id: number; name: string };
export type Settings = Record<string, Record<string, number> | number>;

export type FrameQueueStatusResponse = {
  dataset_id: number;
  status: 'IDLE' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'ERROR';
  total_frames: number;
  processed_frames: number;
  progress_percent: number;
  current_frame_id: number | null;
  current_frame_name: string | null;
  flagged_frames_count: number;
  recent_audited_frames: Array<{
    sample_id: number;
    file_name: string;
    qc_score: number;
    severity: string;
    total_issues: number;
  }>;
  error_message: string | null;
};

export type CVATSyncResponse = {
  status: string;
  message: string;
  dataset_id: number;
  updated_frames_count: number;
  auto_resolved_issues_count: number;
  remaining_issues_count: number;
  synced_at: string;
};

export type CVATVerifyFrameResponse = {
  sample_id: number;
  frame_number: number | null;
  is_verified: boolean;
  resolved_issues_count: number;
  remaining_issues_count: number;
  message: string;
  issues: QCIssue[];
};

export type CVATPushResponse = {
  status: string;
  message: string;
  cvat_task_id?: number | null;
  cvat_job_id?: number | null;
  pushed_shapes_count: number;
  cvat_url?: string | null;
  target_source?: string;
  mode?: string;
};

// --- Multi-Model Ensemble & WBF Types ---

export type DatasetModelsResponse = {
  dataset_id: number;
  models: string[];
  active_models?: string[];
  counts: Record<string, number>;
  has_ensemble: boolean;
};

export type EnsembleTierBreakdown = {
  tier_1_high_consensus: number;
  tier_2_class_disagreement: number;
  tier_3_model_divergence: number;
};

export type EnsembleResponse = {
  status: string;
  message?: string;
  ensemble_source_model?: string;
  models_used?: string[];
  total_fused_predictions?: number;
  tier_breakdown?: EnsembleTierBreakdown;
  files_processed?: number;
  audit_summary?: QCAuditReport;
};

export type EnsembleRunRequest = {
  model_names?: string[];
  model_weights?: Record<string, number>;
  iou_threshold?: number;
  confidence_threshold?: number;
  high_consensus_conf?: number;
  high_consensus_iou?: number;
  ensemble_source_model?: string;
  run_audit?: boolean;
};

export type TriadRunRequest = {
  models?: string[];
  weights?: Record<string, number>;
  conf_thresh?: number;
  iou_thresh?: number;
  sample_limit?: number | null;
  auto_ensemble?: boolean;
};

export type TriadRunResponse = {
  success: boolean;
  skipped_frames?: string[];
  frames_processed: number;
  selected_models: string[];
  model_counts: Record<string, number>;
  total_predictions: number;
  ensemble_result?: EnsembleResponse;
  triad_specs?: Record<string, { label: string; family: string; architecture: string; default_weight: number; description: string }>;
};
