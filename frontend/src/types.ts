export type Feature = {
  value: number | null;
  raw: unknown;
  available: boolean;
  reason_if_unavailable: string | null;
};
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
  qc_score?: number;
  qc_severity?: 'HIGH' | 'MEDIUM' | 'LOW' | 'CLEAN';
  qc_issue_count?: number;
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
};
export type Run = {
  id: number;
  config: unknown;
  needs_regeneration: boolean;
  selections: Sample[];
};
export type Reviewer = { id: number; name: string };
export type Settings = Record<string, Record<string, number> | number>;

