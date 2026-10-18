import { useEffect, useState, useMemo, useRef } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Bot,
  Sparkles,
  Check,
  X,
  AlertTriangle,
  Layers,
  Eye,
  EyeOff,
  RotateCcw,
  Upload,
  Play,
  SlidersHorizontal,
  Search,
  CheckCircle2,
  XCircle,
  HelpCircle,
  TrendingDown,
  Percent,
  ScanSearch,
  FileCheck,
  LoaderCircle,
  Filter,
  ExternalLink,
  Link2,
  Workflow,
  ArrowRight,
  Database,
  Cpu,
  MonitorCheck,
  Undo2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Crosshair,
  Move,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Pause,
  Radio,
  FileUp,
  Send,
  MoreHorizontal,
  ChevronDown,
  Settings,
  Zap,
} from "lucide-react";
import { api, json } from "../api";
import { displayLabel, displayMessage } from "../uiText";
import { QCEvaluationPanel } from './QCEvaluationPanel';
import { TrainingPanel } from './TrainingPanel';
import { WBFExperimentPanel } from './WBFExperimentPanel';
import type {
  Dataset,
  YOLOVariant,
  Sample,
  QCAuditReport,
  QCIssue,
  Prediction,
  Annotation,
  FrameQueueStatusResponse,
  CVATSyncResponse,
  CVATVerifyFrameResponse,
  CVATPushResponse,
  DatasetModelsResponse,
  EnsembleResponse,
  EnsembleRunRequest,
  TriadRunRequest,
  TriadRunResponse,
} from "../types";

// ROI Snapshot Component for Issue Cards
function RoiSnapshot({
  imageUrl,
  location,
  imageWidth = 1280,
  imageHeight = 720,
  index,
  issueType,
  onClick,
}: {
  imageUrl: string | null;
  location?: { x1: number; y1: number; x2: number; y2: number };
  imageWidth?: number;
  imageHeight?: number;
  index: number;
  issueType: string;
  onClick: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasError, setHasError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!imageUrl || !location) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const naturalW = img.naturalWidth || imageWidth;
      const naturalH = img.naturalHeight || imageHeight;

      const scaleX = naturalW / imageWidth;
      const scaleY = naturalH / imageHeight;

      const x1 = Math.max(0, location.x1 * scaleX);
      const y1 = Math.max(0, location.y1 * scaleY);
      const x2 = Math.min(naturalW, location.x2 * scaleX);
      const y2 = Math.min(naturalH, location.y2 * scaleY);

      const bw = Math.max(10, x2 - x1);
      const bh = Math.max(10, y2 - y1);

      // 25% contextual padding around the bounding box
      const pad = Math.max(bw, bh) * 0.25;
      const sx = Math.max(0, x1 - pad);
      const sy = Math.max(0, y1 - pad);
      const sw = Math.min(naturalW - sx, bw + pad * 2);
      const sh = Math.min(naturalH - sy, bh + pad * 2);

      canvas.width = 180;
      canvas.height = 180;

      ctx.fillStyle = "#030712";
      ctx.fillRect(0, 0, 180, 180);

      // Draw cropped image region
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, 180, 180);

      // Draw bounding box on the crop
      const bx = ((x1 - sx) / sw) * 180;
      const by = ((y1 - sy) / sh) * 180;
      const bW = (bw / sw) * 180;
      const bH = (bh / sh) * 180;

      ctx.lineWidth = 3;
      ctx.strokeStyle = issueType === "WRONG_CLASS" ? "#c084fc" : "#ef4444";
      if (issueType !== "WRONG_CLASS") {
        ctx.setLineDash([5, 3]);
      } else {
        ctx.setLineDash([]);
      }
      ctx.strokeRect(bx, by, bW, bH);

      setIsLoaded(true);
    };
    img.onerror = () => {
      setHasError(true);
    };
    img.src = imageUrl;
  }, [imageUrl, location, imageWidth, imageHeight, issueType]);

  return (
    <div
      className="qc-roi-snapshot"
      onClick={onClick}
      title="Bấm để xem cận cảnh vật thể này trên ảnh chính"
    >
      {imageUrl && location && !hasError ? (
        <>
          <canvas ref={canvasRef} className="qc-roi-canvas" />
          {!isLoaded && <div className="qc-roi-loading">Đang tải...</div>}
        </>
      ) : (
        <div className="qc-roi-fallback">
          <ScanSearch size={22} />
          <span>Vật thể #{index}</span>
        </div>
      )}
      <div className="qc-roi-badge-overlay">
        <span className={`roi-num-pin ${issueType === "WRONG_CLASS" ? "wrong" : "missing"}`}>
          #{index}
        </span>
        <span className="roi-locate-pill">
          <Crosshair size={11} /> Xem cận cảnh
        </span>
      </div>
    </div>
  );
}

export function ModelQCPage({
  dataset,
  busy,
  act,
  onRefreshDataset,
}: {
  dataset: Dataset;
  busy: boolean;
  act: (msg: string, fn: () => Promise<void>) => Promise<void>;
  onRefreshDataset: () => Promise<void>;
}) {
  const [selectedYoloModel, setSelectedYoloModel] = useState<YOLOVariant>(() => {
    try { return localStorage.getItem('annopilot.yolo-variant') === 'yolo26s' ? 'yolo26s' : 'yolo26n'; }
    catch { return 'yolo26n'; }
  });
  const selectYoloModel = (variant: YOLOVariant) => {
    setSelectedYoloModel(variant);
    try { localStorage.setItem('annopilot.yolo-variant', variant); } catch {}
  };
  const [report, setReport] = useState<QCAuditReport | null>(null);
  const [queue, setQueue] = useState<Sample[]>([]);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const sampleRequestId = useRef(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setErrorMessage] = useState<string>("");
  const setError = (message: string) => setErrorMessage(displayMessage(message));

  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [issueTypeFilter, setIssueTypeFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Visual diff layer toggles
  const [showGT, setShowGT] = useState<boolean>(true);
  const [showPred, setShowPred] = useState<boolean>(true);
  const [showIssues, setShowIssues] = useState<boolean>(true);

  // Canvas layer declutter & model filtering states
  const [selectedPredModelFilter, setSelectedPredModelFilter] = useState<string>("auto");
  const [labelDensityMode, setLabelDensityMode] = useState<"minimal" | "verbose">("minimal");
  const [minPredConf, setMinPredConf] = useState<number>(0.35);
  const [hoveredPredId, setHoveredPredId] = useState<number | null>(null);
  const [hoveredAnnId, setHoveredAnnId] = useState<number | null>(null);
  const [hoveredIssueId, setHoveredIssueId] = useState<number | null>(null);

  // Modals
  const [showBenchmarkModal, setShowBenchmarkModal] = useState<boolean>(false);
  const [benchmarkMissingRatio, setBenchmarkMissingRatio] = useState<number>(0.15);
  const [benchmarkWrongRatio, setBenchmarkWrongRatio] = useState<number>(0.15);

  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);
  const [uploadSourceModel, setUploadSourceModel] = useState<string>("yolo26n");
  useEffect(() => {
    setUploadSourceModel(current => ['yolo26n', 'yolo26s'].includes(current) ? selectedYoloModel : current);
  }, [selectedYoloModel]);

  // CVAT Integration & Architecture Modal states
  const [showArchModal, setShowArchModal] = useState<boolean>(false);
  const [showCVATModal, setShowCVATModal] = useState<boolean>(false);
  const [cvatBaseUrl, setCvatBaseUrl] = useState<string>(dataset.cvat_base_url || "http://localhost:8080");
  const [cvatTaskId, setCvatTaskId] = useState<string>(dataset.cvat_task_id ? String(dataset.cvat_task_id) : "");
  const [cvatJobId, setCvatJobId] = useState<string>(dataset.cvat_job_id ? String(dataset.cvat_job_id) : "");
  const [cvatToken, setCvatToken] = useState<string>(dataset.cvat_token || "");

  useEffect(() => {
    setCvatBaseUrl(dataset.cvat_base_url || "http://localhost:8080");
    setCvatTaskId(dataset.cvat_task_id ? String(dataset.cvat_task_id) : "");
    setCvatJobId(dataset.cvat_job_id ? String(dataset.cvat_job_id) : "");
    setCvatToken(dataset.cvat_token || "");
  }, [dataset]);

  // Incremental Streaming Queue & CVAT Two-Way Sync states (P0)
  const [queueStatus, setQueueStatus] = useState<FrameQueueStatusResponse | null>(null);
  const [isQueueRunning, setIsQueueRunning] = useState<boolean>(false);
  const [showCVATSyncModal, setShowCVATSyncModal] = useState<boolean>(false);
  const [cvatSyncResult, setCvatSyncResult] = useState<CVATSyncResponse | null>(null);
  const [showCVATPushModal, setShowCVATPushModal] = useState<boolean>(false);
  const [cvatPushResult, setCvatPushResult] = useState<CVATPushResponse | null>(null);
  const [cvatPushMode, setCvatPushMode] = useState<"replace" | "create">("create");
  const [cvatPushSource, setCvatPushSource] = useState<"merged" | "predictions" | "annotations">("merged");
  const [verifyingFrameId, setVerifyingFrameId] = useState<number | null>(null);
  const [auditingFrameId, setAuditingFrameId] = useState<number | null>(null);

  // Multi-Model Ensemble states
  const [showEnsembleModal, setShowEnsembleModal] = useState<boolean>(false);
  const [availableModelsData, setAvailableModelsData] = useState<DatasetModelsResponse | null>(null);
  const [selectedEnsembleModels, setSelectedEnsembleModels] = useState<string[]>([]);
  const [modelWeights, setModelWeights] = useState<Record<string, number>>({});
  const [ensembleIouThresh, setEnsembleIouThresh] = useState<number>(0.55);
  const [ensembleConfThresh, setEnsembleConfThresh] = useState<number>(0.30);
  const [ensembleHighConsensusConf, setEnsembleHighConsensusConf] = useState<number>(0.85);
  const [ensembleResult, setEnsembleResult] = useState<EnsembleResponse | null>(null);
  const [newModelFile, setNewModelFile] = useState<File | null>(null);
  const [newModelName, setNewModelName] = useState<string>("");
  const [runningTriad, setRunningTriad] = useState<boolean>(false);
  const [triadLimit, setTriadLimit] = useState<number | null>(10);
  const triadModels = useMemo(() => [selectedYoloModel, 'fasterrcnn_mobilenet_v3', 'rtdetr_l'], [selectedYoloModel]);

  // More tools menu dropdown state
  const [showMoreToolsMenu, setShowMoreToolsMenu] = useState<boolean>(false);
  const moreToolsMenuRef = useRef<HTMLDivElement>(null);

  // Close more tools menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (moreToolsMenuRef.current && !moreToolsMenuRef.current.contains(event.target as Node)) {
        setShowMoreToolsMenu(false);
      }
    };
    if (showMoreToolsMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showMoreToolsMenu]);

  // Polling effect for background frame queue
  useEffect(() => {
    let timer: any = null;
    if (isQueueRunning) {
      timer = setInterval(async () => {
        try {
          const status = await api<FrameQueueStatusResponse>(`/datasets/${dataset.id}/qc/queue/status`);
          setQueueStatus(status);
          if (status.status === "COMPLETED" || status.status === "PAUSED" || status.status === "ERROR") {
            setIsQueueRunning(false);
            void loadQCData();
            void onRefreshDataset();
          }
        } catch (e) {
          console.error("Queue polling error:", e);
        }
      }, 750);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isQueueRunning, dataset.id]);

  // Multi-model inference commits QC per frame; expose those results while it runs.
  useEffect(() => {
    if (!runningTriad) return;
    let active = true;
    const timer = setInterval(async () => {
      try {
        const [liveReport, liveQueue] = await Promise.all([
          api<QCAuditReport>(`/datasets/${dataset.id}/qc/report?compact=true`),
          api<Sample[]>(`/datasets/${dataset.id}/qc/queue?include_clean=true&compact=true`),
        ]);
        if (active) { setReport(liveReport); setQueue(liveQueue); }
      } catch { /* The final inference response reports errors; polling retries. */ }
    }, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [runningTriad, dataset.id]);

  // Sidebar collapse state
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);

  // Focused issue state for visual-diff linking
  const [focusedIssueId, setFocusedIssueId] = useState<number | null>(null);

  useEffect(() => {
    setFocusedIssueId(null);
    setHoveredPredId(null);
    setHoveredAnnId(null);
    setHoveredIssueId(null);
  }, [selectedSample?.id]);

  // Zoom & Pan state for main canvas
  const [zoomScale, setZoomScale] = useState<number>(1);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
    setIsPanning(false);
  }, [selectedSample?.id]);

  // Native wheel zoom listener with passive: false to prevent outer document scrolling
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = -e.deltaY;
      const step = delta > 0 ? 0.25 : -0.25;
      setZoomScale((prev) => {
        const next = Math.min(4.0, Math.max(0.75, Number((prev + step).toFixed(2))));
        if (next <= 1) {
          setPanOffset({ x: 0, y: 0 });
        }
        return next;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
    };
  }, []);

  const handleZoomIn = () => {
    setZoomScale((prev) => Math.min(4.0, Number((prev + 0.25).toFixed(2))));
  };

  const handleZoomOut = () => {
    setZoomScale((prev) => {
      const next = Math.max(0.75, Number((prev - 0.25).toFixed(2)));
      if (next <= 1) setPanOffset({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetZoom = () => {
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleFitZoom = () => {
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (zoomScale > 1 || e.button === 1) {
      const target = e.target as HTMLElement;
      if (target.closest(".interactive")) return;
      setIsPanning(true);
      setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isPanning) {
      setPanOffset({
        x: Math.round(e.clientX - dragStart.x),
        y: Math.round(e.clientY - dragStart.y),
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  // Auto zoom & pan directly to a specific issue's bounding box
  const zoomToIssueBox = (issue: QCIssue) => {
    setFocusedIssueId(issue.id);
    const stage = document.querySelector(".qc-stage-container");
    if (stage) {
      stage.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    const g = issue.location;
    if (!g) return;

    const imgW = selectedSample?.width || 1280;
    const imgH = selectedSample?.height || 720;
    const boxCenterX = (g.x1 + g.x2) / 2;
    const boxCenterY = (g.y1 + g.y2) / 2;

    const targetZoom = 2.2;
    const normOffsetX = (imgW / 2 - boxCenterX) / imgW;
    const normOffsetY = (imgH / 2 - boxCenterY) / imgH;

    const viewport = viewportRef.current;
    const vpW = viewport?.clientWidth || 800;
    const vpH = viewport?.clientHeight || 450;

    const panX = normOffsetX * vpW * targetZoom;
    const panY = normOffsetY * vpH * targetZoom;

    setZoomScale(targetZoom);
    setPanOffset({ x: Math.round(panX), y: Math.round(panY) });
  };

  const scrollToCard = (issueId: number) => {
    setFocusedIssueId(issueId);
    const cardEl = document.getElementById(`qc-issue-card-${issueId}`);
    if (cardEl) {
      cardEl.scrollIntoView({ behavior: "smooth", block: "center" });
      cardEl.classList.add("flash-highlight");
      setTimeout(() => cardEl.classList.remove("flash-highlight"), 1200);
    }
  };

  const scrollToCanvas = (issueId: number) => {
    setFocusedIssueId(issueId);
    const stage = document.querySelector(".qc-stage-container");
    if (stage) {
      stage.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  // Load QC Data
  const loadQCData = async () => {
    const selectionRequest = ++sampleRequestId.current;
    try {
      setLoading(true);
      setError("");
      // Fetch report & queue in parallel
      const [reportData, queueData] = await Promise.all([
        api<QCAuditReport>(`/datasets/${dataset.id}/qc/report?compact=true`),
        api<Sample[]>(`/datasets/${dataset.id}/qc/queue?include_clean=true&compact=true`),
      ]);
      setReport(reportData);
      setQueue(queueData);
      if (queueData.length > 0) {
        // Preserve currently selected sample if it exists in queueData, fetching fresh details
        const currentId = selectedSample?.id;
        const matchingCurrent = currentId ? queueData.find((s) => s.id === currentId) : null;
        if (matchingCurrent) {
          try {
            const detailed = await api<Sample>(`/samples/${currentId}`);
            if (selectionRequest === sampleRequestId.current) setSelectedSample(detailed);
          } catch (error) {
            if (selectionRequest === sampleRequestId.current) setError(error instanceof Error ? error.message : 'Không tải được chi tiết ảnh.');
          }
        } else {
          const firstFlagged = queueData.find((s) => (s.qc_issue_count || 0) > 0) || queueData[0];
          const detailed = await api<Sample>(`/samples/${firstFlagged.id}`);
          if (selectionRequest === sampleRequestId.current) setSelectedSample(detailed);
        }
      } else {
        if (selectionRequest === sampleRequestId.current) setSelectedSample(null);
      }
    } catch (err: any) {
      setError(err?.message || "Không tải được kết quả kiểm tra nhãn");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadQCData();
  }, [dataset.id]);

  // Refresh single sample detail
  const refreshCurrentSample = async (sampleId: number) => {
    const selectionRequest = ++sampleRequestId.current;
    try {
      const detailed = await api<Sample>(`/samples/${sampleId}`);
      if (selectionRequest === sampleRequestId.current) setSelectedSample(detailed);
      // Update in queue list
      const { annotations, predictions, qc_issues: issues = [], ...summary } = detailed;
      const counts = {
        missing: issues.filter(i => i.issue_type === 'MISSING_OBJECT').length,
        wrong: issues.filter(i => i.issue_type === 'WRONG_CLASS').length,
        resolved: issues.filter(i => i.status !== 'PENDING').length,
        total: issues.length,
      };
      setQueue((prev) => prev.map((s) => (s.id === sampleId ? { ...s, ...summary, qc_issue_counts: counts } : s)));
    } catch (e) {
      if (selectionRequest === sampleRequestId.current) setError(e instanceof Error ? e.message : 'Không tải được chi tiết ảnh.');
    }
  };

  // Run Pretrained Model Detection
  const handleRunDetector = () => {
    void act("Đang chạy mô hình và đối chiếu nhãn…", async () => {
      try {
        await api(`/datasets/${dataset.id}/qc/detect?model_name=${selectedYoloModel}`, { method: "POST" });
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Không chạy được mô hình nhận diện");
      }
    });
  };

  // Start Incremental Streaming Queue (P0)
  const handleStartQueue = async (forceRepredict = false) => {
    void act("Đang bắt đầu kiểm tra từng ảnh…", async () => {
      try {
        const res = await api<FrameQueueStatusResponse>(
          `/datasets/${dataset.id}/qc/queue/start`,
          json("POST", {
            conf_thresh: 0.35,
            model_name: selectedYoloModel,
            force_repredict: forceRepredict,
          })
        );
        setQueueStatus(res);
        setIsQueueRunning(true);
      } catch (err: any) {
        setError(err?.message || "Không thể khởi động hàng đợi");
      }
    });
  };

  // Stop/Pause Incremental Queue
  const handleStopQueue = async () => {
    try {
      const res = await api<FrameQueueStatusResponse>(`/datasets/${dataset.id}/qc/queue/stop`, { method: "POST" });
      setQueueStatus(res);
      setIsQueueRunning(false);
      await loadQCData();
    } catch (err: any) {
      setError(err?.message || "Không thể dừng hàng đợi");
    }
  };

  // On-demand single frame audit
  const handleAuditSingleFrame = async (sampleId: number) => {
    setAuditingFrameId(sampleId);
    void act("Đang nhận diện và đối chiếu lại ảnh…", async () => {
      try {
        await api(`/datasets/${dataset.id}/qc/frame/${sampleId}/audit?force_repredict=true&model_name=${selectedYoloModel}`, { method: "POST" });
        await refreshCurrentSample(sampleId);
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Không kiểm tra lại được ảnh");
      } finally {
        setAuditingFrameId(null);
      }
    });
  };

  // Verify and sync single frame against CVAT
  const handleVerifyFrameCVAT = async (sampleId: number) => {
    setVerifyingFrameId(sampleId);
    void act("Đang đối chiếu ảnh với nhãn trên CVAT…", async () => {
      try {
        const res = await api<CVATVerifyFrameResponse>(`/datasets/${dataset.id}/qc/frame/${sampleId}/cvat-verify`, { method: "POST" });
        await refreshCurrentSample(sampleId);
        await loadQCData();
        if (res.is_verified) {
          alert(`🎉 Hoàn tất! ${res.message} (${res.resolved_issues_count} vấn đề đã được tự động xác nhận)`);
        } else {
          alert(`ℹ️ ${res.message}`);
        }
      } catch (err: any) {
        setError(err?.message || "Không đối chiếu được nhãn trên CVAT");
      } finally {
        setVerifyingFrameId(null);
      }
    });
  };

  // Sync annotations from CVAT file (Source of Truth)
  const handleSyncCVATAnnotations = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const fileInput = form.querySelector<HTMLInputElement>('input[type="file"]');
    const formData = new FormData();
    if (fileInput && fileInput.files && fileInput.files.length > 0) {
      formData.append("file", fileInput.files[0]);
    }

    setShowCVATSyncModal(false);
    void act("Đang lấy nhãn mới nhất từ CVAT…", async () => {
      try {
        const response = await fetch(`/api/datasets/${dataset.id}/qc/cvat-sync`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.detail || "Đồng bộ thất bại");
        }
        const data = (await response.json()) as CVATSyncResponse;
        setCvatSyncResult(data);
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Đồng bộ từ CVAT thất bại");
      }
    });
  };

  // Run Synthetic Benchmark
  const handleRunBenchmark = () => {
    setShowBenchmarkModal(false);
    void act("Đang tạo lỗi mô phỏng để thử nghiệm…", async () => {
      try {
        await api(
          `/datasets/${dataset.id}/qc/synthetic-benchmark?missing_ratio=${benchmarkMissingRatio}&wrong_class_ratio=${benchmarkWrongRatio}&source_model=synthetic_oracle_v1`,
          { method: "POST" },
        );
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Không tạo được thử nghiệm mô phỏng");
      }
    });
  };

  // Upload Predictions File
  const handleUploadPredictions = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const fileInput = form.querySelector<HTMLInputElement>('input[type="file"]');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("source_model", uploadSourceModel);

    setShowUploadModal(false);
    void act("Đang nhập dự đoán và đối chiếu nhãn…", async () => {
      try {
        const response = await fetch(`/api/datasets/${dataset.id}/qc/predictions/upload`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.detail || "Không tải tệp lên được");
        }
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Không nhập được dự đoán");
      }
    });
  };

  // Save CVAT Configuration
  const handleSaveCVATConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    setShowCVATModal(false);
    void act("Đang lưu cấu hình CVAT…", async () => {
      try {
        await api(
          `/datasets/${dataset.id}/cvat-config`,
          json("PUT", {
            cvat_base_url: cvatBaseUrl.trim() || "http://localhost:8080",
            cvat_task_id: cvatTaskId.trim() ? parseInt(cvatTaskId.trim(), 10) : null,
            cvat_job_id: cvatJobId.trim() ? parseInt(cvatJobId.trim(), 10) : null,
            cvat_token: cvatToken.trim() || null,
          }),
        );
        await onRefreshDataset();
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Lỗi lưu cấu hình CVAT");
      }
    });
  };

  // Push annotations directly to CVAT via REST API using PAT
  const handlePushToCVAT = async () => {
    setShowCVATPushModal(false);
    void act("Đang đẩy nhãn đã duyệt sang CVAT Server…", async () => {
      try {
        const res = await api<CVATPushResponse>(
          `/datasets/${dataset.id}/qc/cvat-push`,
          json("POST", {
            token: cvatToken.trim() || undefined,
            cvat_base_url: cvatBaseUrl.trim() || undefined,
            cvat_task_id: cvatTaskId.trim() ? parseInt(cvatTaskId.trim(), 10) : undefined,
            cvat_job_id: cvatJobId.trim() ? parseInt(cvatJobId.trim(), 10) : undefined,
            mode: cvatPushMode,
            target_source: cvatPushSource,
          }),
        );
        setCvatPushResult(res);
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Đẩy nhãn sang CVAT thất bại");
      }
    });
  };

  // Open Ensemble Modal and discover models
  const handleOpenEnsembleModal = async () => {
    void act("Đang tải danh sách model trong dataset...", async () => {
      try {
        const data = await api<DatasetModelsResponse>(`/datasets/${dataset.id}/ensemble/models`);
        setAvailableModelsData(data);
        const nonEnsemble = (data.active_models || data.models).filter((m) => !m.startsWith("ensemble_") && !m.startsWith('synthetic'));
        setSelectedEnsembleModels(nonEnsemble);
        const weights: Record<string, number> = {};
        for (const m of data.models) {
          weights[m] = 1.0;
        }
        setModelWeights(weights);
        setShowEnsembleModal(true);
      } catch (err: any) {
        setError(err?.message || "Không thể tải danh sách model");
      }
    });
  };

  // Run WBF Ensemble
  const handleRunEnsemble = async () => {
    if (selectedEnsembleModels.length < 2) {
      setError("Chọn ít nhất 2 mô hình có dự đoán, hoặc dùng Nhận diện và tạo nhãn gợi ý để chạy 3 mô hình có sẵn.");
      return;
    }
    void act("Đang hợp nhất khung bao và phân mức đồng thuận…", async () => {
      try {
        const res = await api<EnsembleResponse>(
          `/datasets/${dataset.id}/ensemble/run`,
          json("POST", {
            model_names: selectedEnsembleModels,
            model_weights: modelWeights,
            iou_threshold: ensembleIouThresh,
            confidence_threshold: ensembleConfThresh,
            high_consensus_conf: ensembleHighConsensusConf,
            ensemble_source_model: "ensemble_wbf_v1",
            run_audit: true,
          }),
        );
        setEnsembleResult(res);
        await onRefreshDataset();
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Không hợp nhất được dự đoán");
      }
    });
  };

  // Upload an additional model prediction file and fuse
  const handleUploadAdditionalModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newModelFile) {
      setError("Chọn tệp JSON chứa dự đoán của mô hình.");
      return;
    }
    const modelName = newModelName.trim() || `model_${Date.now()}`;
    void act(`Đang nhập dự đoán của '${modelName}' và hợp nhất khung bao…`, async () => {
      try {
        const formData = new FormData();
        formData.append("file", newModelFile);
        formData.append("model_name", modelName);
        formData.append("iou_threshold", String(ensembleIouThresh));
        formData.append("confidence_threshold", String(ensembleConfThresh));
        formData.append("ensemble_source_model", "ensemble_wbf_v1");

        const res = await api<EnsembleResponse>(`/datasets/${dataset.id}/ensemble/upload-and-run`, {
          method: "POST",
          body: formData,
        });
        setEnsembleResult(res);
        setNewModelFile(null);
        setNewModelName("");
        // Reload model list
        const modelsData = await api<DatasetModelsResponse>(`/datasets/${dataset.id}/ensemble/models`);
        setAvailableModelsData(modelsData);
        setSelectedEnsembleModels((modelsData.active_models || modelsData.models).filter((m) => !m.startsWith("ensemble_") && !m.startsWith('synthetic')));
        await onRefreshDataset();
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Không nhập và hợp nhất được dự đoán");
      }
    });
  };

  // Run Option B Triad Ensemble (YOLO26n + Faster R-CNN + RT-DETR)
  const handleRunTriadEnsemble = async (customLimit?: number | null) => {
    setRunningTriad(true);
    const limitToUse = customLimit !== undefined ? customLimit : triadLimit;
    void act("Đang nhận diện bằng YOLO26, Faster R-CNN và RT-DETR rồi hợp nhất kết quả…", async () => {
      try {
        const res = await api<TriadRunResponse>(`/datasets/${dataset.id}/ensemble/run-triad`,
          json("POST", {
            models: triadModels,
            weights: {
              [selectedYoloModel]: modelWeights[selectedYoloModel] ?? 1.0,
              fasterrcnn_mobilenet_v3: modelWeights["fasterrcnn_mobilenet_v3"] ?? 1.0,
              rtdetr_l: modelWeights["rtdetr_l"] ?? 1.2,
            },
            conf_thresh: ensembleConfThresh,
            iou_thresh: ensembleIouThresh,
            sample_limit: limitToUse,
            auto_ensemble: true,
          }),
        );
        if (res.ensemble_result) {
          setEnsembleResult(res.ensemble_result);
        }
        if (!res.success) setError(`Chưa nhận diện hết: ${res.skipped_frames?.length || 0} ảnh thiếu dữ liệu gốc.`);
        const modelsData = await api<DatasetModelsResponse>(`/datasets/${dataset.id}/ensemble/models`);
        setAvailableModelsData(modelsData);
        setSelectedEnsembleModels((modelsData.active_models || modelsData.models).filter((m) => !m.startsWith("ensemble_") && !m.startsWith('synthetic')));
        await onRefreshDataset();
        await loadQCData();
        alert(
          (res.success ? `Đã hoàn tất nhận diện và hợp nhất.\n` : `Đã kết thúc lượt chạy; một số ảnh chưa có kết quả đầy đủ.\n`) +
          `• Số ảnh đã xử lý: ${res.frames_processed}\n` +
          `• Tổng số khung bao dự đoán: ${res.total_predictions.toLocaleString()}\n` +
          `  - ${selectedYoloModel.toUpperCase()}: ${res.model_counts?.[selectedYoloModel] || 0}\n` +
          `  - Faster R-CNN: ${res.model_counts?.fasterrcnn_mobilenet_v3 || 0}\n` +
          `  - RT-DETR-l: ${res.model_counts?.rtdetr_l || 0}\n` +
          `• Nhãn gợi ý sau hợp nhất: ${res.ensemble_result?.total_fused_predictions || 0}\n` +
          `Hãy kiểm tra kết quả trước khi dùng làm nhãn chính thức.`
        );
      } catch (err: any) {
        setError(err?.message || "Không hoàn tất được nhận diện bằng ba mô hình");
      } finally {
        setRunningTriad(false);
      }
    });
  };

  // Helper to generate Deep-link to CVAT
  const getSampleCVATUrl = (sample: Sample) => {
    if (sample.cvat_url) return sample.cvat_url;
    const base = (dataset.cvat_base_url || cvatBaseUrl || "http://localhost:8080").replace(/\/$/, "");
    const frame = sample.frame_number ?? (sample.id - 1);
    const taskId = dataset.cvat_task_id || (cvatTaskId.trim() ? parseInt(cvatTaskId.trim(), 10) : null);
    const jobId = dataset.cvat_job_id || (cvatJobId.trim() ? parseInt(cvatJobId.trim(), 10) : null);

    if (taskId) {
      if (jobId) {
        return `${base}/tasks/${taskId}/jobs/${jobId}?frame=${frame}`;
      }
      return `${base}/tasks/${taskId}?frame=${frame}`;
    }
    return `${base}/tasks?frame=${frame}`;
  };

  // Resolve Issue (Supporting CVAT Resolved, Fast 1-Click Accept, False Positive, and Undo)
  const handleResolveIssue = async (
    issueId: number,
    status: "ACCEPTED" | "REJECTED" | "RESOLVED" | "FALSE_POSITIVE" | "PENDING",
    note = "",
    issueObj?: any,
  ) => {
    if (!selectedSample) return;
    const actMsg =
      status === "RESOLVED"
        ? "Đánh dấu đã sửa trên CVAT…"
        : status === "ACCEPTED"
          ? "Tự động áp dụng sửa 1-Click…"
          : status === "FALSE_POSITIVE" || status === "REJECTED"
            ? "Bác bỏ cảnh báo sai của model…"
            : "Đưa về trạng thái chờ xử lý…";

    void act(actMsg, async () => {
      try {
        await api(
          `/qc/issues/${issueId}`,
          json("PUT", {
            status,
            reviewer_note: note,
            sample_id: selectedSample.id,
            issue_type: issueObj?.issue_type,
            suggested_label: issueObj?.suggested_label,
          }),
        );
        // Reload current sample & dataset audit
        await refreshCurrentSample(selectedSample.id);
        const reportData = await api<QCAuditReport>(`/datasets/${dataset.id}/qc/report?compact=true`);
        setReport(reportData);
      } catch (err: any) {
        setError(err?.message || "Không cập nhật được nghi vấn");
        await refreshCurrentSample(selectedSample.id);
      }
    });
  };

  // Batch 1-Click resolve high-consensus issues (drastically minimizes reviewer clicks)
  const handleBatchResolveConsensus = async (tier = 1) => {
    if (!selectedSample) return;
    void act(
      tier === 1
        ? "Đang tự động áp dụng toàn bộ các nghi vấn đạt đồng thuận cao (Tier 1)…"
        : "Đang tự động áp dụng các nghi vấn theo đồng thuận…",
      async () => {
        try {
          await api(
            `/samples/${selectedSample.id}/qc/batch-resolve`,
            json("POST", {
              consensus_tier: tier,
              status: "ACCEPTED",
              reviewer_note: "1-Click áp dụng đồng thuận từ Multi-Model Ensemble (WBF)",
            }),
          );
          await refreshCurrentSample(selectedSample.id);
          const reportData = await api<QCAuditReport>(`/datasets/${dataset.id}/qc/report?compact=true`);
          setReport(reportData);
        } catch (err: any) {
          setError(err?.message || "Lỗi tự động áp dụng đồng thuận");
          await refreshCurrentSample(selectedSample.id);
        }
      },
    );
  };

  // Filtered Queue
  const filteredQueue = useMemo(() => {
    return queue.filter((item) => {
      // Severity Filter
      if (severityFilter !== "ALL" && item.qc_severity !== severityFilter) {
        return false;
      }
      // Issue Type Filter
      if (issueTypeFilter !== "ALL") {
        const hasIssue = item.qc_issue_counts
          ? (issueTypeFilter === 'MISSING_OBJECT' ? item.qc_issue_counts.missing : item.qc_issue_counts.wrong) > 0
          : item.qc_issues?.some((i) => i.issue_type === issueTypeFilter);
        if (!hasIssue) return false;
      }
      // Search query
      if (searchQuery.trim() && !item.file_name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [queue, severityFilter, issueTypeFilter, searchQuery]);

  // Model filter & prediction filtering for selectedSample (decluttering)
  const sampleModels = useMemo(() => {
    if (!selectedSample?.predictions) return [];
    const set = new Set<string>();
    for (const p of selectedSample.predictions) {
      if (p.source_model && !p.source_model.toLowerCase().startsWith('yolov8')) set.add(p.source_model);
    }
    return Array.from(set);
  }, [selectedSample]);

  const activeModel = useMemo(() => {
    if (selectedPredModelFilter !== "auto" && selectedPredModelFilter !== "all") {
      if (sampleModels.includes(selectedPredModelFilter)) {
        return selectedPredModelFilter;
      }
    }
    // Auto mode: prioritize ensemble models first
    const ensembleModel = sampleModels.find((m) => m.startsWith("ensemble_"));
    if (ensembleModel) return ensembleModel;
    if (sampleModels.includes(selectedYoloModel)) return selectedYoloModel;
    return sampleModels[0] || "all";
  }, [selectedPredModelFilter, sampleModels, selectedYoloModel]);

  const displayedPredictions = useMemo(() => {
    if (!selectedSample?.predictions) return [];
    let preds = selectedSample.predictions.filter((p) => !p.source_model?.toLowerCase().startsWith('yolov8') && (p.confidence ?? 1.0) >= minPredConf);
    if (selectedPredModelFilter === "all") {
      return preds;
    }
    return preds.filter((p) => p.source_model === activeModel);
  }, [selectedSample, minPredConf, selectedPredModelFilter, activeModel]);

  return (
    <div className="qc-page">
      {/* Top Header & Action Controls */}
      <div className="qc-header card">
        <div className="qc-title-group">
          <div className="qc-eyebrow">
            <span className="badge badge-accent">AI HỖ TRỢ</span>
            <span>KIỂM TRA NHÃN VẬT THỂ · BDD100K</span>
          </div>
          <h1>
            Kiểm tra nhãn bằng AI: <span className="highlight">Tìm vật thể bị sót và nhãn sai loại</span>
          </h1>
          <p className="muted">
            AI đối chiếu dự đoán với nhãn hiện có để gợi ý chỗ có thể thiếu vật thể hoặc gán sai loại. Mở từng ảnh để xem bằng chứng và quyết định có sửa nhãn hay không.
          </p>
        </div>

        <div className="qc-actions-bar">
          <label style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
            Mô hình
            <select aria-label="Chọn biến thể YOLO26" value={selectedYoloModel}
              disabled={busy || isQueueRunning || runningTriad}
              onChange={e => selectYoloModel(e.target.value as YOLOVariant)}>
              <option value="yolo26n">YOLO26n · Nhanh, dễ bắt đầu</option>
              <option value="yolo26s">YOLO26s · Chi tiết hơn</option>
            </select>
          </label>
          {/* Action 1: Chạy Streaming Queue (Primary) */}
          <button
            className="primary btn-streaming-start"
            disabled={busy || isQueueRunning}
            onClick={() => handleStartQueue(false)}
            title="Dùng mô hình đang chọn để nhận diện từng ảnh, đối chiếu với nhãn hiện có và xếp nghi vấn cần kiểm tra."
          >
            <Radio size={16} /> {isQueueRunning ? "Đang kiểm tra từng ảnh…" : "Kiểm tra nhãn bằng AI"}
          </button>

          {/* Action 2: Multi-Model Ensemble & Pre-labeling (WBF) */}
          <button
            className="secondary btn-ensemble-trigger"
            style={{ borderColor: "#a855f7", color: "#c084fc", background: "rgba(168, 85, 247, 0.08)" }}
            disabled={busy}
            onClick={() => void handleOpenEnsembleModal()}
            title="Cho nhiều mô hình nhận diện ảnh rồi hợp nhất khung bao thành nhãn gợi ý. Bạn cần kiểm tra lại trước khi sử dụng."
          >
            <Layers size={16} /> Tạo nhãn gợi ý (WBF)
          </button>

          {/* Action 3: Đồng bộ từ CVAT (Inbound) */}
          <button
            className="secondary btn-cvat-sync-trigger"
            disabled={busy}
            onClick={() => setShowCVATSyncModal(true)}
            title="Lấy nhãn bạn đã lưu trên CVAT về AnnoPilot rồi đối chiếu lại. Cần cấu hình kết nối CVAT trước."
          >
            <FileUp size={16} /> Đồng bộ từ CVAT
          </button>

          {/* Action 4: Đẩy nhãn sang CVAT (Outbound) */}
          <button
            className="secondary btn-cvat-push-trigger"
            disabled={busy}
            onClick={() => setShowCVATPushModal(true)}
            title="Chọn nguồn nhãn và cách gửi vào công cụ gán nhãn CVAT để tiếp tục chỉnh sửa."
          >
            <Send size={16} /> Gửi nhãn sang CVAT
          </button>

          {/* Action 5: Dropdown Menu Công cụ Nâng cao / Khác */}
          <div className="qc-dropdown-wrap" ref={moreToolsMenuRef}>
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => setShowMoreToolsMenu(!showMoreToolsMenu)}
              title="Các công cụ nâng cao, cấu hình và tính năng thử nghiệm"
              style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}
            >
              <MoreHorizontal size={16} />
              <span>Công cụ khác</span>
              <ChevronDown size={14} style={{ transform: showMoreToolsMenu ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
            </button>

            {showMoreToolsMenu && (
              <div className="qc-dropdown-menu">
                <div className="qc-dropdown-section-title">Dữ liệu và mô hình</div>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowUploadModal(true);
                  }}
                >
                  <Upload size={15} style={{ color: "var(--accent)" }} />
                  <span>Nhập dự đoán từ tệp JSON</span>
                </button>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    handleOpenEnsembleModal();
                  }}
                >
                  <Sparkles size={15} style={{ color: "#c084fc" }} />
                  <span>Chạy nhận diện bằng 3 mô hình</span>
                </button>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    handleRunDetector();
                  }}
                >
                  <Bot size={15} style={{ color: "#38bdf8" }} />
                  <span>Nhận diện toàn bộ ảnh (công cụ cũ)</span>
                </button>

                <div className="qc-dropdown-divider" />
                <div className="qc-dropdown-section-title">Thử nghiệm và đánh giá</div>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowBenchmarkModal(true);
                  }}
                >
                  <Sparkles size={15} style={{ color: "#facc15" }} />
                  <span>Tạo dữ liệu lỗi mô phỏng</span>
                </button>

                <div className="qc-dropdown-divider" />
                <div className="qc-dropdown-section-title">Cấu hình & Hệ thống</div>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowCVATModal(true);
                  }}
                >
                  <Link2 size={15} style={{ color: "#a855f7" }} />
                  <span>Cấu hình kết nối CVAT {dataset.cvat_task_id ? `(#${dataset.cvat_task_id})` : ""}</span>
                </button>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowArchModal(true);
                  }}
                >
                  <Workflow size={15} style={{ color: "#10b981" }} />
                  <span>Cách AnnoPilot kết nối với CVAT</span>
                </button>
              </div>
            )}
          </div>

          {/* Action 6: Nút Refresh icon */}
          <button className="icon-button" title="Tải lại kết quả kiểm tra nhãn" disabled={busy} onClick={() => void loadQCData()}>
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {ensembleResult && (
        <div className="alert info" style={{ background: "rgba(168, 85, 247, 0.12)", border: "1px solid #a855f7", color: "#f3e8ff", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", borderRadius: "8px", padding: "0.875rem 1.25rem", margin: "1rem 0" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <Layers size={20} style={{ color: "#c084fc", flexShrink: 0 }} />
            <div>
              <strong style={{ color: "#d8b4fe", fontSize: "0.95rem" }}>
                Multi-Model WBF: Đã kết hợp thành công {ensembleResult.total_fused_predictions} khung bao từ {ensembleResult.models_used?.join(", ")}
              </strong>
              <div style={{ fontSize: "0.85rem", color: "#e9d5ff", marginTop: "2px" }}>
                Mức 1 · Đồng thuận cao: <strong>{ensembleResult.tier_breakdown?.tier_1_high_consensus || 0}</strong> ({ensembleResult.total_fused_predictions ? Math.round(((ensembleResult.tier_breakdown?.tier_1_high_consensus || 0) / ensembleResult.total_fused_predictions) * 100) : 0}%) ·{" "}
                Mức 2 · Khác loại: <strong>{ensembleResult.tier_breakdown?.tier_2_class_disagreement || 0}</strong> ·{" "}
                Mức 3 · Chưa đủ đồng thuận: <strong>{ensembleResult.tier_breakdown?.tier_3_model_divergence || 0}</strong>
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              className="secondary"
              style={{ padding: "0.4rem 0.8rem", fontSize: "0.85rem", borderColor: "#a855f7", color: "#e9d5ff" }}
              onClick={() => {
                setCvatPushSource("predictions");
                setShowCVATPushModal(true);
              }}
            >
              <Send size={14} /> Gửi nhãn gợi ý sang CVAT
            </button>
            <button
              className="icon-button"
              style={{ width: "28px", height: "28px", padding: 0 }}
              onClick={() => setEnsembleResult(null)}
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {cvatPushResult && (
        <div className="alert success" style={{ background: "rgba(34, 197, 94, 0.12)", border: "1px solid #22c55e", color: "#f0fdf4", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem", borderRadius: "8px", padding: "0.875rem 1.25rem", margin: "1rem 0" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <CheckCircle2 size={20} style={{ color: "#22c55e", flexShrink: 0 }} />
            <div>
              <strong style={{ color: "#86efac", fontSize: "0.95rem" }}>{cvatPushResult.message}</strong>
              <div style={{ fontSize: "0.85rem", color: "#bbf7d0", marginTop: "2px" }}>
                Đã đẩy thành công {cvatPushResult.pushed_shapes_count} khung bao sang tác vụ CVAT #{cvatPushResult.cvat_task_id || dataset.cvat_task_id} (công việc #{cvatPushResult.cvat_job_id || dataset.cvat_job_id}).
              </div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            {cvatPushResult.cvat_url && (
              <a
                href={cvatPushResult.cvat_url}
                target="_blank"
                rel="noreferrer"
                className="btn-cvat-primary"
                style={{ padding: "0.375rem 0.75rem", fontSize: "0.8rem", textDecoration: "none" }}
              >
                <ExternalLink size={14} /> Mở CVAT xem kết quả
              </a>
            )}
            <button className="icon-button" onClick={() => setCvatPushResult(null)} style={{ color: "#bbf7d0" }}>
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="alert error" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
          <button className="secondary" disabled={loading} onClick={() => void loadQCData()}>Tải lại dữ liệu</button>
          <button className="icon-button" aria-label="Đóng thông báo lỗi" onClick={() => setError("")}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Streaming Frame Processing Live Banner */}
      {(isQueueRunning || (queueStatus && queueStatus.status !== "IDLE")) && (
        <div className="qc-streaming-banner">
          <div className="qc-streaming-banner-header">
            <div className="qc-streaming-banner-title">
              <Radio size={18} className={isQueueRunning ? "text-accent animate-pulse" : "text-muted"} />
              <span>Tiến độ kiểm tra từng ảnh</span>
              {queueStatus?.status === "RUNNING" && (
                <span className="qc-streaming-pill-running">
                  <LoaderCircle size={12} className="spin" /> Đang chạy
                </span>
              )}
              {queueStatus?.status === "COMPLETED" && (
                <span className="qc-streaming-pill-completed">
                  <CheckCircle2 size={12} /> Đã hoàn thành ({queueStatus.total_frames} ảnh)
                </span>
              )}
              {queueStatus?.status === "PAUSED" && (
                <span className="qc-streaming-pill-paused">
                  <Pause size={12} /> Tạm dừng
                </span>
              )}
            </div>
            <div className="qc-streaming-actions">
              {isQueueRunning ? (
                <button className="secondary small" onClick={handleStopQueue} title="Tạm dừng hàng đợi">
                  <Pause size={14} /> Tạm dừng
                </button>
              ) : (
                <button className="secondary small" onClick={() => handleStartQueue(false)} title="Tiếp tục xử lý">
                  <Play size={14} /> Tiếp tục
                </button>
              )}
              <button
                className="icon-button"
                onClick={() => setQueueStatus(null)}
                title="Đóng thanh tiến trình"
              >
                <X size={14} />
              </button>
            </div>
          </div>

          {/* Progress Track */}
          <div className="qc-streaming-progress-track">
            <div
              className="qc-streaming-progress-fill"
              style={{ width: `${queueStatus?.progress_percent || 0}%` }}
            />
          </div>

          <div className="qc-streaming-banner-meta">
            <div>
              <span>Tiến độ: <strong>{queueStatus?.processed_frames || 0}</strong> / {queueStatus?.total_frames || 0} ảnh ({queueStatus?.progress_percent || 0}%)</span>
              {queueStatus?.current_frame_name && (
                <span style={{ marginLeft: "1rem" }}>
                  Đang xử lý: <code style={{ color: "#38bdf8" }}>{queueStatus.current_frame_name}</code>
                </span>
              )}
            </div>
            <div>
              <span>Phát hiện nghi vấn: <strong style={{ color: "#f87171" }}>{queueStatus?.flagged_frames_count || 0} ảnh</strong></span>
            </div>
          </div>
        </div>
      )}

      {/* CVAT Sync Result Alert Banner */}
      {cvatSyncResult && (
        <div className="qc-sync-alert-banner">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <CheckCircle2 size={20} />
            <span>{cvatSyncResult.message}</span>
          </div>
          <button className="icon-button" onClick={() => setCvatSyncResult(null)}>
            <X size={14} />
          </button>
        </div>
      )}

      <section className="card beginner-guide" aria-labelledby="qc-start-title">
        <h2 id="qc-start-title">Bạn muốn làm gì?</h2>
        <div className="beginner-paths">
          <div><strong>Đã có nhãn, muốn tìm lỗi</strong><p>Bấm <b>Kiểm tra nhãn bằng AI</b>, chọn một ảnh trong danh sách và đối chiếu vùng nghi vấn.</p></div>
          <div><strong>Ảnh chưa có nhãn</strong><p>Mở <b>Tạo nhãn gợi ý (WBF)</b>, chạy nhận diện rồi gửi nhãn gợi ý sang CVAT để kiểm tra và sửa.</p></div>
          <div><strong>Đã sửa nhãn trên CVAT</strong><p>Lưu trên CVAT, sau đó bấm <b>Đồng bộ từ CVAT</b> để cập nhật nhãn và kiểm tra lại nghi vấn.</p></div>
        </div>
        <p className="muted">Chưa cần huấn luyện để bắt đầu: bạn có thể dùng các mô hình có sẵn. Điểm nghi vấn giúp chọn ảnh cần xem trước, không phải xác suất nhãn sai.</p>
      </section>
      <TrainingPanel key={`training-${dataset.id}`} datasetId={dataset.id} datasetName={dataset.name}
        onCheckpointChanged={async model => {
          if (model === 'yolo26n' || model === 'yolo26s') selectYoloModel(model);
          await loadQCData();
        }} />
      <QCEvaluationPanel key={dataset.id} datasetId={dataset.id} />
      <details className="card">
        <summary><strong>Luồng dữ liệu CVAT ↔ AnnoPilot</strong></summary>
        <p><strong>Lần đầu:</strong> Xuất tác vụ từ CVAT theo định dạng CVAT for images kèm ảnh, rồi dùng Thêm bộ dữ liệu. Cấu hình địa chỉ CVAT, mã tác vụ (Task ID) và mã truy cập (PAT) trong Công cụ khác. Chỉ nhập mã tác vụ chưa đủ để tải ảnh về.</p>
        <p><strong>Lấy nhãn về:</strong> Lưu các thay đổi trên CVAT, rồi bấm Đồng bộ từ CVAT và để trống phần chọn tệp. Nếu chọn tệp XML / JSON, ứng dụng chỉ đọc bản nhãn trong tệp, chưa kiểm tra nhãn hiện tại trên CVAT.</p>
        <p><strong>Gửi nhãn đi:</strong> Sau khi tạo nhãn gợi ý, bấm Gửi nhãn sang CVAT và chọn nguồn Nhãn gợi ý từ AI. Kiểm tra, sửa và lưu trên CVAT, rồi đồng bộ về. Nút Áp dụng gợi ý tại AnnoPilot chỉ sửa bản nhãn trong ứng dụng này.</p>
        <p><strong>Chọn cách gửi:</strong> Thêm khung bao mới giữ nguyên khung cũ, kể cả khung đang sai loại. Thay thế khung bao sẽ thay các khung 2D trên ảnh thuộc bộ dữ liệu; cần lấy nhãn mới nhất từ CVAT về trước. Để trống Job ID nếu muốn dùng toàn tác vụ.</p>
      </details>
      {report && ((report.unprocessed_samples_count || 0) > 0 || (report.inference_error_samples_count || 0) > 0) && (
        <p role="status" className="card text-warning" style={{ padding: '1rem' }}>
          Độ phủ chưa đầy đủ: {report.unprocessed_samples_count || 0} ảnh chưa có dự đoán; {report.inference_error_samples_count || 0} ảnh bị lỗi nhận diện. Chưa thể kết luận chất lượng các ảnh này.
        </p>
      )}
      {/* KPI Metrics Dashboard Cards */}
      {report && (
        <div className="grid four qc-kpi-grid">
          {/* Workload Reduction */}
          <div className="card qc-kpi-card highlight-kpi">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Ảnh chưa có cảnh báo</span>
              <TrendingDown size={20} className="text-success" />
            </div>
            <div className="qc-kpi-value">
              {report.workload_reduction_percentage.toFixed(1)}%
            </div>
            <div className="qc-meter-bar">
              <div
                className="qc-meter-fill"
                style={{ width: `${Math.min(100, report.workload_reduction_percentage)}%` }}
              />
            </div>
            <p className="qc-kpi-note">
              <strong>{report.clean_samples_count}</strong> / {report.total_samples} ảnh chưa có cảnh báo. Vẫn cần kiểm tra để xác nhận chất lượng nhãn.
            </p>
          </div>

          {/* Flagged Samples */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Ảnh cần kiểm tra</span>
              <ShieldAlert size={20} className="text-warning" />
            </div>
            <div className="qc-kpi-value text-warning">
              {report.flagged_samples_count}
              <small> / {report.total_samples}</small>
            </div>
            <div className="qc-pill-row">
              <span className="pill pill-high">Cao: {report.severity_breakdown.HIGH || 0}</span>
              <span className="pill pill-med">Vừa: {report.severity_breakdown.MEDIUM || 0}</span>
              <span className="pill pill-low">Thấp: {report.severity_breakdown.LOW || 0}</span>
            </div>
            <p className="qc-kpi-note">Ảnh có điểm nghi vấn cao được xếp trước.</p>
          </div>

          {/* Missing Objects */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Có thể thiếu vật thể</span>
              <ScanSearch size={20} className="text-accent" />
            </div>
            <div className="qc-kpi-value text-accent">
              {report.total_missing_candidates}
            </div>
            <p className="qc-kpi-note">AI phát hiện vật thể chưa có khung bao trong nhãn hiện tại.</p>
          </div>

          {/* Wrong Classes */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Có thể sai loại vật thể</span>
              <AlertTriangle size={20} className="text-purple" />
            </div>
            <div className="qc-kpi-value text-purple">
              {report.total_wrong_class_candidates}
            </div>
            <p className="qc-kpi-note">Khung bao gần trùng nhau nhưng AI và nhãn hiện tại ghi loại vật thể khác nhau.</p>
          </div>
        </div>
      )}

      {/* Main Split Layout: Left Queue & Right Visual Diff / Closed-loop review */}
      <div className={`qc-main-grid ${isSidebarOpen ? "" : "sidebar-collapsed"}`}>
        {/* LEFT COLUMN: PRIORITIZED QUEUE (collapsible sidebar) */}
        <section className={`card qc-queue-card ${isSidebarOpen ? "" : "collapsed"}`}>
          {/* Sidebar Toggle Button */}
          <button
            className="sidebar-toggle-btn"
            onClick={() => setIsSidebarOpen((v) => !v)}
            title={isSidebarOpen ? "Ẩn danh sách ảnh" : "Hiện danh sách ảnh"}
          >
            {isSidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
            {!isSidebarOpen && (
              <span className="sidebar-collapsed-label">
                {filteredQueue.length} ảnh
              </span>
            )}
          </button>
          {isSidebarOpen && (
            <div className="qc-queue-header">
              <div>
                <h3>Danh sách ảnh cần kiểm tra</h3>
                <p className="muted small">Điểm nghi vấn cao nhất được xếp trên cùng</p>
              </div>
              <span className="badge badge-primary">{filteredQueue.length} ảnh</span>
            </div>
          )}

          {/* Filter Bar */}
          {isSidebarOpen && (
            <div className="qc-filter-controls">
              <div className="search-wrap">
                <Search size={14} />
                <input
                  type="text"
                  placeholder="Tìm theo tên file…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <button className="icon-button" onClick={() => setSearchQuery("")}>
                    <X size={12} />
                  </button>
                )}
              </div>

              <div className="qc-filter-pills">
                {["ALL", "HIGH", "MEDIUM", "LOW", "CLEAN"].map((lvl) => (
                  <button
                    key={lvl}
                    className={`pill-btn ${severityFilter === lvl ? "active" : ""}`}
                    onClick={() => setSeverityFilter(lvl)}
                  >
                    {lvl === 'ALL' ? 'Tất cả' : displayLabel(lvl)}
                  </button>
                ))}
              </div>

              <div className="qc-filter-pills">
                <button
                  className={`pill-btn ${issueTypeFilter === "ALL" ? "active" : ""}`}
                  onClick={() => setIssueTypeFilter("ALL")}
                >
                  Mọi loại lỗi
                </button>
                <button
                  className={`pill-btn ${issueTypeFilter === "MISSING_OBJECT" ? "active" : ""}`}
                  onClick={() => setIssueTypeFilter("MISSING_OBJECT")}
                >
                  Chỉ sót vật thể
                </button>
                <button
                  className={`pill-btn ${issueTypeFilter === "WRONG_CLASS" ? "active" : ""}`}
                  onClick={() => setIssueTypeFilter("WRONG_CLASS")}
                >
                  Chỉ sai loại vật thể
                </button>
              </div>
            </div>
          )}

          {/* Queue List */}
          {isSidebarOpen && (
            <div className="qc-queue-list">
              {loading ? (
                <div className="empty">
                  <LoaderCircle className="spin" size={28} />
                  <p>Đang tải hàng đợi QC…</p>
                </div>
              ) : filteredQueue.length === 0 ? (
                <div className="empty">
                  <FileCheck size={28} />
                  <p>Không có mẫu nào phù hợp với bộ lọc.</p>
                </div>
              ) : (
                filteredQueue.map((item, idx) => {
                  const isSelected = selectedSample?.id === item.id;
                  const issues = item.qc_issues || [];
                  const missingCount = item.qc_issue_counts?.missing ?? issues.filter((i) => i.issue_type === "MISSING_OBJECT").length;
                  const wrongCount = item.qc_issue_counts?.wrong ?? issues.filter((i) => i.issue_type === "WRONG_CLASS").length;
                  const resolvedCount = item.qc_issue_counts?.resolved ?? issues.filter((i) => i.status !== "PENDING").length;
                  const issueCount = item.qc_issue_counts?.total ?? issues.length;

                  return (
                    <div
                      key={item.id}
                      className={`qc-queue-item ${isSelected ? "selected" : ""} ${item.qc_severity ? item.qc_severity.toLowerCase() : ""
                        }`}
                      onClick={() => refreshCurrentSample(item.id)}
                    >
                      <div className="qc-item-top">
                        <span className="qc-item-rank">#{idx + 1}</span>
                        <span
                          className={`badge qc-score-badge ${item.qc_severity ? item.qc_severity.toLowerCase() : ""
                            }`}
                        >
                          Điểm nghi vấn: {(item.qc_score || 0).toFixed(3)}
                        </span>
                        <span className={`badge ${item.qc_severity ? item.qc_severity.toLowerCase() : "clean"}`}>
                          {item.qc_processing_state === 'ERROR' ? 'Lỗi nhận diện' : item.qc_processing_state === 'NOT_RUN' ? 'Chưa chạy nhận diện' : displayLabel(item.qc_severity || 'CLEAN')}
                        </span>
                      </div>

                      <div className="qc-item-filename" title={item.file_name}>
                        {item.file_name}
                      </div>

                      <div className="qc-item-chips">
                        <span className="chip chip-frame">
                          Ảnh #{item.frame_number ?? (item.id - 1)}
                        </span>
                        <a
                          href={getSampleCVATUrl(item)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="queue-cvat-link"
                          title="Mở ảnh này trên CVAT"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink size={11} /> CVAT
                        </a>
                        {missingCount > 0 && (
                          <span className="chip chip-missing">+{missingCount} Sót</span>
                        )}
                        {wrongCount > 0 && (
                          <span className="chip chip-wrong">+{wrongCount} Sai loại</span>
                        )}
                        {issueCount > 0 && resolvedCount === issueCount && (
                          <span className="chip chip-resolved">
                            <Check size={10} /> Đã sửa hết
                          </span>
                        )}
                        {issueCount === 0 && (
                          <span className="chip chip-clean">
                            <ShieldCheck size={10} /> Chưa có nghi vấn
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </section>

        {/* RIGHT COLUMN: DUAL-OVERLAY VISUAL DIFF & RESOLUTION */}
        <section className="card qc-inspection-card">
          {selectedSample ? (
            <div className="qc-inspection-wrap">
              {/* Inspection Top Bar */}
              <div className="qc-inspection-header">
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                  <div>
                    <div className="eyebrow">
                      MẪU #{selectedSample.id} · ẢNH #{selectedSample.frame_number ?? (selectedSample.id - 1)} · {selectedSample.task_type}
                    </div>
                    <h2>{selectedSample.file_name}</h2>
                  </div>
                  <a
                    href={getSampleCVATUrl(selectedSample)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-cvat-primary"
                    title="Mở đúng ảnh này trên CVAT để xem và sửa nhãn"
                  >
                    <ExternalLink size={16} />
                    <span>Sửa trong CVAT (ảnh #{selectedSample.frame_number ?? (selectedSample.id - 1)})</span>
                  </a>
                  <button
                    type="button"
                    className="secondary"
                    style={{ padding: "0.45rem 0.85rem", fontSize: "0.85rem", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                    disabled={busy || auditingFrameId === selectedSample.id}
                    onClick={() => handleAuditSingleFrame(selectedSample.id)}
                    title="Chạy nhận diện lại ảnh này rồi đối chiếu với nhãn hiện tại"
                  >
                    <RefreshCw size={14} className={auditingFrameId === selectedSample.id ? "spin" : ""} />
                    <span>Kiểm tra lại ảnh này</span>
                  </button>
                </div>

                {/* Layer Toggles & Smart Filters */}
                <div className="qc-layer-toggles" style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                  <span className="toggle-label" style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--muted)" }}>Lớp:</span>
                  <button
                    className={`toggle-btn ${showGT ? "active gt" : ""}`}
                    onClick={() => setShowGT(!showGT)}
                    title="Ẩn hoặc hiện nhãn hiện có để đối chiếu với AI"
                  >
                    <span className="color-dot gt" /> Nhãn người ({selectedSample.annotations?.length || 0})
                  </button>
                  <button
                    className={`toggle-btn ${showPred ? "active pred" : ""}`}
                    onClick={() => setShowPred(!showPred)}
                    title="Ẩn hoặc hiện khung bao do AI dự đoán"
                  >
                    <span className="color-dot pred" /> AI ({displayedPredictions.length})
                  </button>
                  <button
                    className={`toggle-btn ${showIssues ? "active issue" : ""}`}
                    onClick={() => setShowIssues(!showIssues)}
                    title="Bật/Tắt vùng nghi vấn lỗi QC"
                  >
                    <span className="color-dot issue" /> Vùng nghi vấn ({selectedSample.qc_issues?.length || 0})
                  </button>

                  {/* Model Source Filter Dropdown */}
                  {showPred && sampleModels.length > 0 && (
                    <div style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem", marginLeft: "0.25rem" }}>
                      <Layers size={13} style={{ color: "#c084fc" }} />
                      <select
                        value={selectedPredModelFilter === "auto" ? activeModel : selectedPredModelFilter}
                        onChange={(e) => setSelectedPredModelFilter(e.target.value)}
                        className="qc-model-select"
                        style={{
                          padding: "0.3rem 0.6rem",
                          fontSize: "0.8rem",
                          borderRadius: "6px",
                          background: activeModel.startsWith("ensemble_") ? "rgba(168, 85, 247, 0.18)" : "rgba(255, 255, 255, 0.06)",
                          border: `1px solid ${activeModel.startsWith("ensemble_") ? "rgba(168, 85, 247, 0.5)" : "rgba(255, 255, 255, 0.18)"}`,
                          color: activeModel.startsWith("ensemble_") ? "#d8b4fe" : "var(--text)",
                          fontWeight: 600,
                        }}
                        title="Chọn mô hình có khung bao bạn muốn xem trên ảnh"
                      >
                        {sampleModels.some((m) => m.startsWith("ensemble_")) && (
                          <option value={sampleModels.find((m) => m.startsWith("ensemble_"))}>
                            Chỉ hiện kết quả đã hợp nhất từ nhiều mô hình
                          </option>
                        )}
                        {sampleModels
                          .filter((m) => !m.startsWith("ensemble_"))
                          .map((m) => (
                            <option key={m} value={m}>
                              Mô hình: {m}
                            </option>
                          ))}
                        <option value="all">Tất cả mô hình ({selectedSample.predictions?.length || 0} khung bao · chi tiết kỹ thuật)</option>
                      </select>
                    </div>
                  )}

                  {/* Label Density Mode Toggle */}
                  <div style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", marginLeft: "0.25rem" }}>
                    <button
                      type="button"
                      className={`toggle-btn ${labelDensityMode === "minimal" ? "active" : ""}`}
                      style={{
                        padding: "0.28rem 0.65rem",
                        fontSize: "0.78rem",
                        background: labelDensityMode === "minimal" ? "rgba(56, 189, 248, 0.15)" : "transparent",
                        borderColor: labelDensityMode === "minimal" ? "rgba(56, 189, 248, 0.45)" : "rgba(255, 255, 255, 0.15)",
                        color: labelDensityMode === "minimal" ? "#38bdf8" : "var(--muted)",
                        fontWeight: 600,
                      }}
                      onClick={() => setLabelDensityMode(labelDensityMode === "minimal" ? "verbose" : "minimal")}
                      title={labelDensityMode === "minimal" ? "Đang bật chế độ Gọn gàng (Chỉ hiện chữ khi rê chuột) - Bấm để hiện tất cả chữ" : "Đang hiện tất cả chữ - Bấm để bật chế độ Gọn gàng"}
                    >
                      <Eye size={12} />
                      <span>{labelDensityMode === "minimal" ? "Nhãn: Gọn (Rê chuột)" : "Nhãn: Hiện tất cả"}</span>
                    </button>
                  </div>

                  {/* Min Confidence Threshold Selector */}
                  {showPred && (
                    <div style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem", marginLeft: "0.25rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Độ tin cậy tối thiểu:</span>
                      <select
                        value={String(minPredConf)}
                        onChange={(e) => setMinPredConf(parseFloat(e.target.value))}
                        style={{
                          padding: "0.25rem 0.5rem",
                          fontSize: "0.78rem",
                          borderRadius: "4px",
                          background: "rgba(0, 0, 0, 0.4)",
                          border: "1px solid rgba(255, 255, 255, 0.15)",
                          color: "var(--text)",
                        }}
                        title="Ẩn dự đoán có độ tin cậy thấp hơn ngưỡng đã chọn"
                      >
                        <option value="0.25">25% (Rộng)</option>
                        <option value="0.35">35% (Chuẩn)</option>
                        <option value="0.50">50% (Cao)</option>
                        <option value="0.65">65% (Rất cao)</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>

              {/* Visual Diff Canvas Stage with Zoom & Pan */}
              <div className="qc-stage-container">
                {/* Floating Zoom & Pan Controls Toolbar */}
                <div className="qc-canvas-toolbar">
                  <div className="toolbar-group zoom-controls">
                    <button
                      type="button"
                      className="zoom-tool-btn"
                      onClick={handleZoomOut}
                      disabled={zoomScale <= 0.75}
                      title="Thu nhỏ ảnh (-)"
                    >
                      <ZoomOut size={14} />
                    </button>
                    <span className="zoom-percentage-badge">{Math.round(zoomScale * 100)}%</span>
                    <button
                      type="button"
                      className="zoom-tool-btn"
                      onClick={handleZoomIn}
                      disabled={zoomScale >= 4.0}
                      title="Phóng to ảnh (+)"
                    >
                      <ZoomIn size={14} />
                    </button>
                    <button
                      type="button"
                      className={`zoom-tool-btn text-btn ${zoomScale === 1 && panOffset.x === 0 && panOffset.y === 0 ? "active" : ""}`}
                      onClick={handleResetZoom}
                      title="Khôi phục kích thước ban đầu (100%)"
                    >
                      <RotateCcw size={12} /> 100%
                    </button>
                    <button
                      type="button"
                      className="zoom-tool-btn text-btn"
                      onClick={handleFitZoom}
                      title="Căn vừa khung nhìn"
                    >
                      <Maximize2 size={12} /> Vừa khung
                    </button>
                  </div>

                  <div className="toolbar-group zoom-status-hint">
                    {zoomScale > 1 ? (
                      <span className="pan-active-hint">
                        <Move size={12} /> Kéo để di chuyển ảnh · Cuộn chuột để phóng to / thu nhỏ
                      </span>
                    ) : (
                      <span className="zoom-default-hint">
                        Cuộn chuột để phóng to · Bấm ảnh thu nhỏ của nghi vấn để xem cận cảnh
                      </span>
                    )}
                  </div>
                </div>

                {/* Viewport for Zoom & Pan */}
                <div
                  className={`qc-canvas-viewport ${isPanning ? "is-panning" : ""}`}
                  ref={viewportRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                >
                  <div
                    className="qc-canvas-stage"
                    style={{
                      aspectRatio: `${selectedSample.width || 1280}/${selectedSample.height || 720}`,
                      transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomScale})`,
                      transformOrigin: "center center",
                      transition: isPanning ? "none" : "transform 0.18s cubic-bezier(0.16, 1, 0.3, 1)",
                      cursor: zoomScale > 1 ? (isPanning ? "grabbing" : "grab") : "default",
                    }}
                  >
                    {selectedSample.media_url ? (
                      <img src={selectedSample.media_url} alt={selectedSample.file_name} draggable={false} />
                    ) : (
                      <div className="blueprint-canvas">
                        <div className="blueprint-grid" />
                        <div className="blueprint-text">
                          <span>Không có ảnh gốc kèm theo · Hiển thị toạ độ không gian chuẩn hoá BDD100K</span>
                        </div>
                      </div>
                    )}

                    {/* SVG Overlay Layer */}
                    <svg
                      viewBox={`0 0 ${selectedSample.width || 1280} ${selectedSample.height || 720}`}
                      preserveAspectRatio="none"
                    >
                      {/* 1. Ground Truth Annotations (Solid Green/Cyan) */}
                      {showGT &&
                        selectedSample.annotations?.map((ann) => {
                          const g = ann.geometry;
                          if (!g || ann.shape_type !== "BBOX_2D") return null;
                          const isHovered = hoveredAnnId === ann.id;
                          const isDimmed = (hoveredAnnId !== null && !isHovered) || (focusedIssueId !== null);
                          const showLabel = labelDensityMode === "verbose" || isHovered;
                          return (
                            <g
                              key={`gt-${ann.id}`}
                              className={`svg-box gt-box interactive ${isHovered ? "is-hovered" : ""} ${isDimmed ? "is-dimmed" : ""}`}
                              onMouseEnter={() => setHoveredAnnId(ann.id)}
                              onMouseLeave={() => setHoveredAnnId(null)}
                            >
                              <title>{`Nhãn người: ${ann.label}`}</title>
                              <rect
                                x={g.x1}
                                y={g.y1}
                                width={g.x2 - g.x1}
                                height={g.y2 - g.y1}
                              />
                              {showLabel && (
                                <g className="gt-label-chip">
                                  <rect
                                    x={g.x1}
                                    y={Math.max(2, g.y1 - 18)}
                                    width={Math.max(68, (ann.label.length + 8) * 7.2)}
                                    height={18}
                                    rx={3}
                                    fill="rgba(15, 23, 42, 0.9)"
                                    stroke="#2dd4bf"
                                    strokeWidth={1}
                                  />
                                  <text
                                    x={g.x1 + 5}
                                    y={Math.max(14, g.y1 - 5)}
                                    fill="#2dd4bf"
                                    fontSize={11}
                                    fontWeight={700}
                                  >
                                    Người: {ann.label}
                                  </text>
                                </g>
                              )}
                            </g>
                          );
                        })}

                      {/* 2. Model Predictions (Filtered & Density Controlled) */}
                      {showPred &&
                        displayedPredictions.map((pred) => {
                          const g = pred.geometry;
                          if (!g) return null;
                          const isEnsemble = pred.source_model?.startsWith("ensemble_") || false;
                          const isHovered = hoveredPredId === pred.id;
                          const isDimmed = (hoveredPredId !== null && !isHovered) || (focusedIssueId !== null);
                          const showLabel = labelDensityMode === "verbose" || isHovered;
                          const labelText = `${isEnsemble ? "⭐ " : ""}${pred.label} (${(pred.confidence * 100).toFixed(0)}%) · ${pred.source_model || "model"}`;
                          const badgeW = Math.max(80, labelText.length * 6.8 + 12);
                          const badgeH = 18;
                          const badgeY = Math.min((selectedSample.height || 720) - badgeH - 2, g.y2 + 4);
                          const badgeX = Math.max(2, Math.min(g.x1, (selectedSample.width || 1280) - badgeW - 2));

                          return (
                            <g
                              key={`pred-${pred.id}`}
                              className={`svg-box pred-box interactive ${isEnsemble ? "ensemble" : ""} ${isHovered ? "is-hovered" : ""} ${isDimmed ? "is-dimmed" : ""}`}
                              onMouseEnter={() => setHoveredPredId(pred.id)}
                              onMouseLeave={() => setHoveredPredId(null)}
                            >
                              <title>{`Model (${pred.source_model || "unknown"}): ${pred.label} (${(pred.confidence * 100).toFixed(0)}%)`}</title>
                              <rect
                                x={g.x1}
                                y={g.y1}
                                width={g.x2 - g.x1}
                                height={g.y2 - g.y1}
                              />
                              {showLabel && (
                                <g className="pred-label-chip">
                                  <rect
                                    x={badgeX}
                                    y={badgeY}
                                    width={badgeW}
                                    height={badgeH}
                                    rx={3}
                                    fill="rgba(15, 23, 42, 0.92)"
                                    stroke={isEnsemble ? "#c084fc" : "#fb923c"}
                                    strokeWidth={1}
                                  />
                                  <text
                                    x={badgeX + 6}
                                    y={badgeY + 13}
                                    fill={isEnsemble ? "#e9d5ff" : "#fed7aa"}
                                    fontSize={11}
                                    fontWeight={700}
                                  >
                                    {labelText}
                                  </text>
                                </g>
                              )}
                            </g>
                          );
                        })}

                      {/* 3. Flagged Issue Highlights (Glowing Red / Magenta / Purple) */}
                      {showIssues &&
                        (() => {
                          const issues = selectedSample.qc_issues || [];
                          const indexed = issues.map((issue, idx) => ({ issue, idx }));
                          // Render focused or hovered issue last so it stays on top in SVG
                          const sorted = [...indexed].sort((a, b) => {
                            const aActive = a.issue.id === focusedIssueId || a.issue.id === hoveredIssueId;
                            const bActive = b.issue.id === focusedIssueId || b.issue.id === hoveredIssueId;
                            if (aActive && !bActive) return 1;
                            if (!aActive && bActive) return -1;
                            return 0;
                          });

                          return sorted.map(({ issue, idx }) => {
                            const g = issue.location;
                            if (!g) return null;
                            const isResolved = issue.status !== "PENDING";
                            const isFocused = focusedIssueId === issue.id || hoveredIssueId === issue.id;
                            const isDimmed = (focusedIssueId !== null || hoveredIssueId !== null) && !isFocused;
                            const isWrongClass = issue.issue_type === "WRONG_CLASS";
                            const boxW = Math.max(1, g.x2 - g.x1);
                            const boxH = Math.max(1, g.y2 - g.y1);

                            const labelText = isWrongClass
                              ? `#${idx + 1} [⇄] ${issue.human_label || '?'} ➔ ${issue.suggested_label}${issue.evidence.iou != null ? ` (IoU ${(issue.evidence.iou * 100).toFixed(0)}%)` : ''}`
                              : `#${idx + 1} [!] Sót: ${issue.suggested_label} (${Math.round((issue.evidence.model_confidence || 0) * 100)}%)`;

                            const charW = 7.4;
                            const badgeW = Math.max(80, labelText.length * charW + 18);
                            const badgeH = 20;
                            const badgeY = Math.max(badgeH + 4, g.y1 - 6);
                            const badgeX = Math.max(2, Math.min(g.x1, (selectedSample.width || 1280) - badgeW - 4));

                            // In minimal mode: wide label banner ONLY shows when hovered or focused
                            const showDetailedBanner = labelDensityMode === "verbose" || isFocused;

                            return (
                              <g
                                key={`issue-${issue.id}`}
                                className={`svg-box issue-highlight interactive ${isWrongClass ? "wrong-class" : "missing-object"
                                  } ${isResolved ? "resolved" : ""} ${isFocused ? "is-focused" : ""
                                  } ${isDimmed ? "is-dimmed" : ""}`}
                                onMouseEnter={() => {
                                  setFocusedIssueId(issue.id);
                                  setHoveredIssueId(issue.id);
                                }}
                                onMouseLeave={() => {
                                  setFocusedIssueId(null);
                                  setHoveredIssueId(null);
                                }}
                                onClick={() => scrollToCard(issue.id)}
                              >
                                <title>{`Nghi vấn #${idx + 1}: ${labelText}\nBấm để cuộn xuống thẻ chi tiết`}</title>

                                {/* Bounding box rectangle */}
                                <rect
                                  className="issue-rect"
                                  x={g.x1 - 2}
                                  y={g.y1 - 2}
                                  width={boxW + 4}
                                  height={boxH + 4}
                                />

                                {/* Corner HUD reticles when focused */}
                                {isFocused && (
                                  <g className="hud-corners">
                                    <path d={`M ${g.x1 - 8} ${g.y1 + 8} L ${g.x1 - 8} ${g.y1 - 8} L ${g.x1 + 8} ${g.y1 - 8}`} className="hud-corner-line" />
                                    <path d={`M ${g.x2 - 8} ${g.y1 - 8} L ${g.x2 + 8} ${g.y1 - 8} L ${g.x2 + 8} ${g.y1 + 8}`} className="hud-corner-line" />
                                    <path d={`M ${g.x1 - 8} ${g.y2 - 8} L ${g.x1 - 8} ${g.y2 + 8} L ${g.x1 + 8} ${g.y2 + 8}`} className="hud-corner-line" />
                                    <path d={`M ${g.x2 - 8} ${g.y2 + 8} L ${g.x2 + 8} ${g.y2 + 8} L ${g.x2 + 8} ${g.y2 - 8}`} className="hud-corner-line" />
                                  </g>
                                )}

                                {/* Pill Badge Background & Text (conditional in minimal mode to prevent clutter) */}
                                {showDetailedBanner && (
                                  <>
                                    <rect
                                      className="issue-badge-bg"
                                      x={badgeX}
                                      y={badgeY - badgeH + 4}
                                      width={badgeW}
                                      height={badgeH}
                                      rx={4}
                                      ry={4}
                                    />
                                    <text
                                      className="issue-badge-text"
                                      x={badgeX + 8}
                                      y={badgeY}
                                    >
                                      {labelText}
                                    </text>
                                  </>
                                )}

                                {/* Corner Number Pin Marker (ALWAYS visible for clean and concise referencing) */}
                                <circle
                                  className="issue-pin-circle"
                                  cx={g.x1}
                                  cy={Math.max(12, g.y1)}
                                  r={isFocused ? 13 : 10}
                                />
                                <text
                                  className="issue-pin-text"
                                  x={g.x1}
                                  y={Math.max(12, g.y1) + 3.5}
                                  textAnchor="middle"
                                >
                                  {idx + 1}
                                </text>
                              </g>
                            );
                          });
                        })()}
                    </svg>
                  </div>
                </div>

                <div className="qc-stage-legend">
                  <span>
                    <i className="legend-dot gt" /> Nhãn hiện có
                  </span>
                  <span>
                    <i className="legend-dot pred" /> AI ({activeModel.startsWith("ensemble_") ? "⭐ Ensemble WBF" : activeModel})
                  </span>
                  <span>
                    <i className="legend-dot wrong-class" /> [⇄] Có thể sai loại (tím)
                  </span>
                  <span>
                    <i className="legend-dot missing" /> [!] Sót vật thể (Đỏ)
                  </span>
                  <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "var(--muted)" }}>
                    Rê chuột vào khung bao để xem nhãn · Bấm số nghi vấn để xem cách xử lý
                  </span>
                </div>
              </div>

              {/* Closed-Loop Resolution Issues Cards */}
              <div className="qc-issues-section">
                <div className="qc-issues-heading">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                    <h3>
                      Các nghi vấn trên ảnh này ({selectedSample.qc_issues?.length || 0} nghi vấn)
                    </h3>
                  </div>
                  <p className="muted small">
                    Số #1, #2… của mỗi nghi vấn tương ứng với khung bao trên ảnh. Đối chiếu nhãn với ảnh gốc, rồi chọn cách xử lý bên dưới.
                  </p>
                </div>

                {/* Multi-Model Consensus Auto-Resolution Banner (drastically reduces human clicks) */}
                {(() => {
                  const pendingIssues = selectedSample.qc_issues?.filter((i) => i.status === "PENDING") || [];
                  const tier1Issues = pendingIssues.filter(
                    (i) => i.evidence?.consensus_tier === 1
                  );
                  if (tier1Issues.length === 0) return null;

                  return (
                    <div
                      className="qc-consensus-resolve-banner"
                      style={{
                        background: "linear-gradient(135deg, rgba(168, 85, 247, 0.16) 0%, rgba(99, 102, 241, 0.16) 100%)",
                        border: "1px solid rgba(168, 85, 247, 0.45)",
                        borderRadius: "10px",
                        padding: "0.85rem 1.25rem",
                        marginTop: "0.75rem",
                        marginBottom: "1rem",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: "0.85rem",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                        <div style={{ background: "rgba(168, 85, 247, 0.25)", padding: "0.5rem", borderRadius: "8px", color: "#c084fc", display: "flex", alignItems: "center" }}>
                          <Sparkles size={20} />
                        </div>
                        <div>
                          <div style={{ fontWeight: 700, color: "#f3e8ff", fontSize: "0.95rem" }}>
                            Có {tier1Issues.length} / {pendingIssues.length} nghi vấn đồng thuận cao (mức 1)
                          </div>
                          <div style={{ fontSize: "0.82rem", color: "#d8b4fe", marginTop: "2px" }}>
                            Đồng thuận chỉ là tín hiệu ưu tiên. Reviewer cần kiểm tra bằng chứng trước khi chấp nhận các gợi ý.
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="primary"
                        style={{
                          background: "linear-gradient(135deg, #a855f7 0%, #6366f1 100%)",
                          border: "none",
                          color: "#ffffff",
                          padding: "0.5rem 1.15rem",
                          fontWeight: 700,
                          fontSize: "0.85rem",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.45rem",
                          borderRadius: "6px",
                          boxShadow: "0 2px 10px rgba(168, 85, 247, 0.35)",
                          cursor: "pointer",
                        }}
                        disabled={busy}
                        onClick={() => handleBatchResolveConsensus(1)}
                        title="Chỉ áp dụng các gợi ý mức đồng thuận cao sau khi bạn đã xem ảnh. Thay đổi được lưu trong AnnoPilot."
                      >
                        <Zap size={15} />
                        <span>Đã kiểm tra: chấp nhận gợi ý ({tier1Issues.length})</span>
                      </button>
                    </div>
                  );
                })()}

                {/* Quick Navigation Strip for dense issue frames */}
                {selectedSample.qc_issues && selectedSample.qc_issues.length > 1 && (
                  <div className="qc-issue-nav-strip">
                    <span className="nav-strip-label">Nghi vấn ({selectedSample.qc_issues.length}):</span>
                    <div className="nav-strip-scroll">
                      {selectedSample.qc_issues.map((iss, i) => (
                        <button
                          key={iss.id}
                          className={`nav-issue-chip ${iss.issue_type === "MISSING_OBJECT" ? "missing" : "wrong-class"} ${focusedIssueId === iss.id ? "active" : ""
                            } ${iss.status !== "PENDING" ? "resolved" : ""}`}
                          onMouseEnter={() => setFocusedIssueId(iss.id)}
                          onMouseLeave={() => setFocusedIssueId(null)}
                          onClick={() => scrollToCard(iss.id)}
                        >
                          <span className="chip-num">#{i + 1}</span>
                          <span className="chip-text">
                            {iss.issue_type === "MISSING_OBJECT" ? `Sót: ${iss.suggested_label}` : `${iss.human_label || '?'} ➔ ${iss.suggested_label}`}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {(!selectedSample.qc_issues || selectedSample.qc_issues.length === 0) ? (
                  <div className="qc-clean-banner">
                    <ShieldCheck size={28} className="text-success" />
                    <div>
                      <h4>Ảnh chưa có nghi vấn</h4>
                      <p className="muted">
                        Ảnh này chưa có cảnh báo. Cần chắc chắn AI đã nhận diện ảnh và kiểm tra thủ công trước khi kết luận nhãn đúng.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="qc-issues-list">
                    {selectedSample.qc_issues.map((issue, idx) => {
                      const isPending = issue.status === "PENDING";
                      const isResolved = issue.status === "RESOLVED";
                      const isAccepted = issue.status === "ACCEPTED";
                      const isRejected = issue.status === "REJECTED" || issue.status === "FALSE_POSITIVE";
                      const isFocused = focusedIssueId === issue.id;
                      const g = issue.location;

                      return (
                        <div
                          key={issue.id}
                          id={`qc-issue-card-${issue.id}`}
                          className={`qc-issue-card ${issue.issue_type === "MISSING_OBJECT" ? "missing" : "wrong-class"
                            } ${!isPending ? "resolved" : ""} ${isFocused ? "focused-card" : ""
                            }`}
                          onMouseEnter={() => setFocusedIssueId(issue.id)}
                          onMouseLeave={() => setFocusedIssueId(null)}
                        >
                          <div className="qc-issue-header">
                            <div className="qc-issue-type-badge">
                              {/* Prominent Number Badge matching canvas pin */}
                              <span className={`qc-index-badge ${issue.issue_type === "MISSING_OBJECT" ? "missing" : "wrong-class"}`}>
                                #{idx + 1}
                              </span>
                              {issue.issue_type === "MISSING_OBJECT" ? (
                                <>
                                  <ScanSearch size={16} />
                                  <strong>CÓ THỂ THIẾU VẬT THỂ</strong>
                                </>
                              ) : (
                                <>
                                  <AlertTriangle size={16} />
                                  <strong>CÓ THỂ SAI LOẠI VẬT THỂ</strong>
                                </>
                              )}
                              {g && (
                                <span className="qc-box-coords-chip" title="Tọa độ Bounding Box [x1, y1 · width × height]">
                                  Khung [{Math.round(g.x1)}, {Math.round(g.y1)} · {Math.round(g.x2 - g.x1)}×{Math.round(g.y2 - g.y1)}px]
                                </span>
                              )}
                            </div>

                            <div className="qc-issue-badges">
                              {issue.evidence?.consensus_tier === 1 && (
                                <span
                                  className="badge"
                                  style={{ background: "rgba(168, 85, 247, 0.22)", color: "#d8b4fe", border: "1px solid rgba(168, 85, 247, 0.55)", fontWeight: 700 }}
                                  title={issue.evidence.tier_reason || "Đồng thuận cao giữa các kiến trúc mô hình (WBF)"}
                                >
                                  Đồng thuận cao · {issue.evidence.num_models_agreed ?? 'Nhiều'} mô hình
                                </span>
                              )}
                              {issue.evidence?.consensus_tier === 2 && (
                                <span
                                  className="badge"
                                  style={{ background: "rgba(245, 158, 11, 0.22)", color: "#fcd34d", border: "1px solid rgba(245, 158, 11, 0.55)", fontWeight: 700 }}
                                  title={issue.evidence.tier_reason || "Các mô hình nhận diện loại vật thể khác nhau"}
                                >
                                  Khác loại vật thể · {issue.evidence.num_models_agreed ?? 'Nhiều'} mô hình
                                </span>
                              )}
                              <span className="badge badge-accent">
                                Điểm nghi vấn: {issue.qc_score.toFixed(3)}
                              </span>
                              <span
                                className={`badge ${isPending
                                    ? "badge-pending"
                                    : isResolved
                                      ? "badge-resolved"
                                      : isAccepted
                                        ? "badge-accepted"
                                        : "badge-false-positive"
                                  }`}
                              >
                                {issue.status}
                              </span>
                            </div>
                          </div>

                          {/* Comparison Visual Panel with Object Snapshot */}
                          <div className="qc-comparison-box">
                            {/* Object RoI Crop Snapshot */}
                            <div className="cmp-snapshot-col">
                              <RoiSnapshot
                                imageUrl={selectedSample.media_url}
                                location={g}
                                imageWidth={selectedSample.width || 1280}
                                imageHeight={selectedSample.height || 720}
                                index={idx + 1}
                                issueType={issue.issue_type}
                                onClick={() => zoomToIssueBox(issue)}
                              />
                            </div>

                            <div className="cmp-col human">
                              <span className="cmp-label">Nhãn Người Gán</span>
                              <div className="cmp-val">
                                {issue.human_label ? (
                                  <span className="tag tag-human">{issue.human_label}</span>
                                ) : (
                                  <span className="tag tag-empty">Chưa gán nhãn</span>
                                )}
                              </div>
                            </div>

                            <div className="cmp-divider">
                              <span>so với</span>
                            </div>

                            <div className="cmp-col model">
                              <span className="cmp-label">AI đề xuất</span>
                              <div className="cmp-val">
                                <span className="tag tag-model">
                                  {issue.suggested_label}
                                </span>
                                <small className="conf-text">
                                  Độ tin cậy: {((issue.evidence.model_confidence || 0) * 100).toFixed(1)}%
                                  {issue.evidence.iou != null && ` · IoU: ${issue.evidence.iou.toFixed(2)}`}
                                </small>
                              </div>
                            </div>
                          </div>

                          {/* Evidence Reason */}
                          <div className="qc-reason-box">
                            <HelpCircle size={15} />
                            <span>{issue.evidence.reason}</span>
                          </div>
                          {issue.evidence.per_model_predictions?.length > 0 && (
                            <details className="qc-model-evidence">
                              <summary>Bằng chứng từng mô hình · {issue.evidence.detection_count ?? issue.evidence.contributing_models?.length} mô hình phát hiện</summary>
                              {issue.evidence.per_model_predictions.map((p: any, index: number) => (
                                <p key={index}><strong>{p.model_id}</strong>: {p.label} · độ tin cậy {(p.confidence * 100).toFixed(1)}% · khung [{p.geometry.x1}, {p.geometry.y1}, {p.geometry.x2}, {p.geometry.y2}]</p>
                              ))}
                              {!!issue.evidence.absent_models?.length && <p>Không có detection tương ứng: {issue.evidence.absent_models.join(', ')}</p>}
                              <p className="muted">Độ tin cậy của mô hình chưa phải xác suất nhãn đúng. Bạn vẫn cần xem ảnh để xác nhận.</p>
                            </details>
                          )}

                          {/* Actions Bar */}
                          <div className="qc-issue-actions">
                            {isPending ? (
                              <>
                                <a
                                  href={issue.cvat_url || getSampleCVATUrl(selectedSample)}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="btn-cvat-primary"
                                  title="Mở đúng ảnh này trên CVAT để sửa nhãn"
                                >
                                  <ExternalLink size={15} />
                                  Mở trong CVAT (ảnh #{issue.frame_number ?? (selectedSample.frame_number ?? (selectedSample.id - 1))})
                                </a>
                                <button
                                  className="btn-verify-frame-cvat"
                                  disabled={busy || verifyingFrameId === selectedSample.id}
                                  onClick={() => handleVerifyFrameCVAT(selectedSample.id)}
                                  title="Đọc nhãn đã lưu trên CVAT và đối chiếu lại để gỡ các nghi vấn đã sửa"
                                >
                                  <RefreshCw size={14} className={verifyingFrameId === selectedSample.id ? "spin" : ""} />
                                  Kiểm tra & Đồng bộ từ CVAT
                                </button>
                                <button
                                  className="btn-resolve-cvat"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "RESOLVED", "Ghi nhận đã sửa, chưa xác nhận trực tiếp từ CVAT", issue)}
                                  title="Chỉ ghi nhận rằng bạn đã sửa. Dùng Kiểm tra & Đồng bộ từ CVAT để xác nhận nhãn thực tế."
                                >
                                  <CheckCircle2 size={15} />
                                  Đã sửa (ghi nhận thủ công)
                                </button>
                                <button
                                  className="btn-accept"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "ACCEPTED", "Áp dụng gợi ý vào nhãn trong AnnoPilot", issue)}
                                  title="Thêm khung bao hoặc sửa loại theo gợi ý AI trong AnnoPilot. Chưa gửi thay đổi sang CVAT."
                                >
                                  <Sparkles size={15} />
                                  Áp dụng gợi ý tại AnnoPilot
                                </button>
                                <button
                                  className="btn-reject"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "FALSE_POSITIVE", "Báo sai của model", issue)}
                                  title="Đánh dấu AI cảnh báo nhầm và giữ nhãn hiện tại"
                                >
                                  <X size={15} />
                                  Bác bỏ cảnh báo sai
                                </button>
                              </>
                            ) : isResolved ? (
                              <div className="resolution-status resolved">
                                <CheckCircle2 size={18} />
                                <span>
                                  Đã ghi nhận xử lý. Xem ghi chú để biết kết quả được xác nhận trực tiếp từ CVAT hay chỉ ghi nhận thủ công / nhập từ tệp.
                                </span>
                                <button
                                  className="btn-undo"
                                  title="Hoàn tác về trạng thái chờ xử lý"
                                  onClick={() => handleResolveIssue(issue.id, "PENDING", "", issue)}
                                >
                                  <RotateCcw size={12} /> Hoàn tác
                                </button>
                              </div>
                            ) : isAccepted ? (
                              <div className="resolution-status accepted">
                                <CheckCircle2 size={18} />
                                <span>
                                  Đã cập nhật nhãn trong AnnoPilot. Muốn cập nhật CVAT, hãy dùng Gửi nhãn sang CVAT.
                                </span>
                                <button
                                  className="btn-undo"
                                  title="Hoàn tác về trạng thái chờ xử lý"
                                  onClick={() => handleResolveIssue(issue.id, "PENDING")}
                                >
                                  <RotateCcw size={12} /> Hoàn tác
                                </button>
                              </div>
                            ) : (
                              <div className="resolution-status false-positive">
                                <XCircle size={18} />
                                <span>
                                  Đã bác bỏ cảnh báo. Nhãn hiện tại được giữ nguyên.
                                </span>
                                <button
                                  className="btn-undo"
                                  title="Hoàn tác về trạng thái chờ xử lý"
                                  onClick={() => handleResolveIssue(issue.id, "PENDING")}
                                >
                                  <RotateCcw size={12} /> Hoàn tác
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="empty">
              <ScanSearch size={36} />
              <h3>Chọn một ảnh trong hàng đợi để tiến hành QC</h3>
              <p className="muted">
                Các ảnh có nguy cơ lỗi cao nhất được xếp ở đầu danh sách bên trái.
              </p>
            </div>
          )}
        </section>
      </div>

      {/* MODAL: BENCHMARK SIMULATION */}
      {showBenchmarkModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">MÔ PHỎNG KIỂM THỬ TẬP TRUNG</div>
                <h2>Tạo lỗi mô phỏng để thử nghiệm</h2>
              </div>
              <button className="icon-button" aria-label="Đóng thử nghiệm mô phỏng" onClick={() => setShowBenchmarkModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Tự động mô phỏng các ca sót vật thể và tráo nhãn (theo phân phối chuẩn của đề tài BDD100K) để kiểm thử
              đầu-cuối toàn bộ luồng phát hiện và đo lường Workload Reduction.
            </p>

            <div className="form-group">
              <label>
                Tỷ lệ mô phỏng sót vật thể (Missing Ratio): <strong>{(benchmarkMissingRatio * 100).toFixed(0)}%</strong>
                <input
                  type="range"
                  min="0.05"
                  max="0.40"
                  step="0.05"
                  value={benchmarkMissingRatio}
                  onChange={(e) => setBenchmarkMissingRatio(parseFloat(e.target.value))}
                />
              </label>
            </div>

            <div className="form-group">
              <label>
                Tỷ lệ mô phỏng gán sai nhãn (Wrong Class Ratio): <strong>{(benchmarkWrongRatio * 100).toFixed(0)}%</strong>
                <input
                  type="range"
                  min="0.05"
                  max="0.40"
                  step="0.05"
                  value={benchmarkWrongRatio}
                  onChange={(e) => setBenchmarkWrongRatio(parseFloat(e.target.value))}
                />
              </label>
            </div>

            <div className="modal-actions">
              <button className="secondary" onClick={() => setShowBenchmarkModal(false)}>
                Hủy bỏ
              </button>
              <button className="primary" onClick={handleRunBenchmark}>
                <Sparkles size={16} /> Bắt đầu mô phỏng
              </button>
            </div>
          </section>
        </div>
      )}

      {/* MODAL: UPLOAD PREDICTIONS */}
      {showUploadModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">TÍCH HỢP MÔ HÌNH BÊN NGOÀI</div>
                <h2>Nhập dự đoán của mô hình</h2>
              </div>
              <button className="icon-button" aria-label="Đóng cửa sổ nhập dự đoán" onClick={() => setShowUploadModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Nạp kết quả dự đoán từ YOLO26, Faster R-CNN hoặc bất kỳ detector nào theo định dạng JSON chuẩn BDD100K.
            </p>

            <form onSubmit={handleUploadPredictions}>
              <div className="form-group">
                <label>
                  Tên mô hình nhận diện:
                  <input
                    type="text"
                    value={uploadSourceModel}
                    onChange={(e) => setUploadSourceModel(e.target.value)}
                    required
                  />
                </label>
              </div>

              <div className="form-group">
                <label className="file-drop">
                  <Upload size={24} />
                  <strong>Tệp dự đoán JSON</strong>
                  <span>Tệp JSON cần có danh sách frames / predictions theo định dạng hỗ trợ</span>
                  <input type="file" accept=".json" required />
                </label>
              </div>

              <div className="modal-actions">
                <button type="button" className="secondary" onClick={() => setShowUploadModal(false)}>
                  Hủy bỏ
                </button>
                <button type="submit" className="primary">
                  <Upload size={16} /> Tải lên & Đối soát
                </button>
              </div>
            </form>
          </section>
        </div>
      )}

      {/* MODAL: CVAT SERVER CONFIGURATION */}
      {showCVATModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">TÍCH HỢP HỆ THỐNG</div>
                <h2>Kết nối với CVAT</h2>
              </div>
              <button className="icon-button" aria-label="Đóng cấu hình CVAT" onClick={() => setShowCVATModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Nhập thông tin CVAT của nhóm để dùng nút
              <code> [Mở trong CVAT] </code> để mở đúng ảnh cần sửa trong trình chỉnh sửa nhãn.
            </p>

            <form onSubmit={handleSaveCVATConfig}>
              <div className="form-group">
                <label>
                  <span>Địa chỉ máy chủ CVAT:</span>
                  <input
                    type="url"
                    placeholder="http://localhost:8080"
                    value={cvatBaseUrl}
                    onChange={(e) => setCvatBaseUrl(e.target.value)}
                    required
                  />
                  <small className="muted">Ví dụ: http://localhost:8080 hoặc địa chỉ CVAT của nhóm</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>Mã tác vụ CVAT (Task ID):</span>
                  <input
                    type="number"
                    placeholder="VD: 1, 42"
                    value={cvatTaskId}
                    onChange={(e) => setCvatTaskId(e.target.value)}
                  />
                  <small className="muted">Lấy số tác vụ trong đường dẫn CVAT, ví dụ /tasks/12 thì nhập 12. Tác vụ phải chứa đúng ảnh của bộ dữ liệu này.</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>Mã công việc CVAT (Job ID, không bắt buộc):</span>
                  <input
                    type="number"
                    placeholder="VD: 1"
                    value={cvatJobId}
                    onChange={(e) => setCvatJobId(e.target.value)}
                  />
                  <small className="muted">Để trống để đồng bộ toàn tác vụ. Chỉ nhập khi tất cả ảnh thuộc cùng một công việc CVAT; mã công việc khác mã phân đoạn.</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>Mã truy cập CVAT (PAT):</span>
                  <input
                    type="password"
                    placeholder=""
                    value={cvatToken}
                    onChange={(e) => setCvatToken(e.target.value)}
                  />
                  <small className="muted">Tạo mã truy cập cá nhân trong tài khoản CVAT, rồi dán vào đây để ứng dụng được phép đọc và gửi nhãn.</small>
                </label>
              </div>

              <div className="qc-reason-box" style={{ margin: "1rem 0" }}>
                <ExternalLink size={16} />
                <span>
                  Định dạng đường dẫn mở ảnh:{" "}
                  <code>
                    {cvatBaseUrl || "http://localhost:8080"}/tasks/{cvatTaskId || "{task_id}"}
                    {cvatJobId ? `/jobs/${cvatJobId}` : ""}?frame={"{"}frame_number{"}"}
                  </code>
                </span>
              </div>

              <div className="modal-actions">
                <button type="button" className="secondary" onClick={() => setShowCVATModal(false)}>
                  Hủy bỏ
                </button>
                <button type="submit" className="primary">
                  <Check size={16} /> Lưu cấu hình
                </button>
              </div>
            </form>
          </section>
        </div>
      )}

      {/* MODAL: CVAT X QC DASHBOARD ARCHITECTURE DIAGRAM */}
      {showArchModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal qc-arch-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">THIẾT KẾ HỆ THỐNG · ĐỀ TÀI N2-04D</div>
                <h2>Sơ đồ kiến trúc CVAT × QC Dashboard</h2>
              </div>
              <button className="icon-button" aria-label="Đóng sơ đồ kết nối" onClick={() => setShowArchModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Chuẩn hóa theo tài liệu thiết kế <em>CVAT × QC Dashboard v0.1</em>. Phân tách rành mạch giữa nơi lưu trữ/chỉnh sửa dữ liệu và nơi xếp hạng, đối soát chất lượng.
            </p>

            <div className="arch-flow-container">
              {/* TIER 1: CVAT */}
              <div className="arch-tier-card tier-cvat">
                <div className="arch-tier-header">
                  <div className="arch-tier-title">
                    <Database size={18} className="text-accent" />
                    <span>CVAT · Nơi lưu và sửa nhãn gốc</span>
                  </div>
                  <span className="arch-tier-tag">Nơi sửa dữ liệu</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Video và ảnh gốc</strong>
                    <p>Lưu trữ hình ảnh gốc theo Frame Index (0, 1, 2... N)</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Trình chỉnh sửa nhãn</strong>
                    <p>Vẽ khung bao và chọn loại vật thể</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Kết nối và xuất dữ liệu CVAT</strong>
                    <p>Xuất dữ liệu định dạng COCO / BDD100K JSON</p>
                  </div>
                </div>
              </div>

              {/* CONNECTOR 1 */}
              <div className="arch-connector">
                <div className="arch-connector-badge">
                  <ArrowRight size={14} />
                  <span>Xuất nhãn Ground Truth (BDD100K / COCO format)</span>
                  <ArrowRight size={14} />
                </div>
              </div>

              {/* TIER 2: QC ENGINE & PIPELINE */}
              <div className="arch-tier-card tier-engine">
                <div className="arch-tier-header">
                  <div className="arch-tier-title">
                    <Cpu size={18} className="text-purple" />
                    <span>Bộ xử lý kiểm tra nhãn</span>
                  </div>
                  <span className="arch-tier-tag">Tầng đối soát tự động</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Mô hình nhận diện đã học sẵn</strong>
                    <p>Dự đoán vị trí và loại vật thể theo bộ loại BDD100K</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Bộ đối chiếu khác biệt</strong>
                    <p>So mức trùng khớp khung bao để tìm vật thể thiếu và nhãn sai loại</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Xếp hạng nghi vấn</strong>
                    <p>Tính QC Risk Score, đẩy ca bất đồng lên đầu hàng đợi</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Tạo đường dẫn tới ảnh cần sửa</strong>
                    <p>Tạo đường dẫn tới đúng ảnh: <code>/tasks/ID/jobs/ID?frame=N</code></p>
                  </div>
                </div>
              </div>

              {/* CONNECTOR 2 */}
              <div className="arch-connector">
                <div className="arch-connector-badge">
                  <ArrowRight size={14} />
                  <span>Nút bấm [ Open in CVAT ] mở đúng Frame trên CVAT Editor để sửa</span>
                  <ArrowRight size={14} />
                </div>
              </div>

              {/* TIER 3: QC DASHBOARD */}
              <div className="arch-tier-card tier-dashboard">
                <div className="arch-tier-header">
                  <div className="arch-tier-title">
                    <MonitorCheck size={18} className="text-success" />
                    <span>Giao diện người kiểm tra</span>
                  </div>
                  <span className="arch-tier-tag">Tầng ra quyết định & Review</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Tổng quan kết quả kiểm tra</strong>
                    <p>Thống kê ảnh chưa có cảnh báo; cần kiểm chứng chất lượng riêng</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Danh sách ưu tiên kiểm tra</strong>
                    <p>Xếp hạng danh sách ảnh từ nguy cơ cao nhất xuống thấp</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>So sánh nhãn và dự đoán trên ảnh</strong>
                    <p>Ẩn hoặc hiện nhãn hiện có, dự đoán AI và vùng nghi vấn</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Ghi nhận cách xử lý</strong>
                    <p>Đánh dấu <em>Đã sửa trên CVAT</em>, <em>1-Click Sửa tự động</em>, hoặc <em>Cảnh báo sai</em></p>
                  </div>
                </div>
              </div>
            </div>

            <div className="arch-footer-callout">
              <strong>Nguyên tắc vận hành:</strong> Dùng AnnoPilot để chọn ảnh cần kiểm tra trước, rồi bấm <code>Mở trong CVAT</code> để sửa nhãn gốc trên CVAT. Lưu thay đổi, sau đó đồng bộ lại về AnnoPilot.
            </div>

            <div className="modal-actions">
              <button type="button" className="primary" onClick={() => setShowArchModal(false)}>
                Đã hiểu
              </button>
            </div>
          </section>
        </div>
      )}

      {/* MODAL: CVAT TWO-WAY SYNCHRONIZATION (P0: System of Record) */}
      {showCVATSyncModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">ĐỒNG BỘ 2-CHIỀU VỚI CVAT</div>
                <h2>Lấy nhãn mới nhất từ CVAT</h2>
              </div>
              <button className="icon-button" aria-label="Đóng cửa sổ lấy nhãn từ CVAT" onClick={() => setShowCVATSyncModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              CVAT là nguồn dữ liệu chuẩn (Single Source of Truth). Hệ thống sẽ kết nối trực tiếp tới CVAT REST API (sử dụng PAT) để kéo toàn bộ nhãn mới nhất vừa sửa và lưu trên CVAT, đối soát lại và tự động gỡ các cảnh báo đã được sửa.
            </p>

            <form onSubmit={handleSyncCVATAnnotations}>
              <div className="qc-reason-box" style={{ margin: "0 0 1rem 0" }}>
                <Database size={16} />
                <span>
                  Nguồn đồng bộ trực tiếp: <strong>{cvatBaseUrl}</strong> · Task ID:{" "}
                  <strong>{cvatTaskId || dataset.cvat_task_id || "Chưa cấu hình"}</strong>
                  {cvatJobId || dataset.cvat_job_id ? ` · Job ID: ${cvatJobId || dataset.cvat_job_id}` : ""}
                </span>
              </div>

              <div className="form-group">
                <label>
                  <span>Hoặc nhập bản nhãn từ tệp (CVAT XML / BDD100K JSON):</span>
                  <input type="file" accept=".xml,.json" />
                </label>
                <small className="muted">
                  Để trống để lấy nhãn mới nhất trực tiếp từ CVAT. Nếu chọn tệp, ứng dụng chỉ nhập bản nhãn trong tệp đó.
                </small>
              </div>

              <div className="modal-actions">
                <button type="button" className="secondary" onClick={() => setShowCVATSyncModal(false)}>
                  Hủy bỏ
                </button>
                <button type="submit" className="primary btn-cvat-sync-trigger">
                  <RefreshCw size={16} /> Đồng bộ nhãn mới từ CVAT
                </button>
              </div>
            </form>
          </section>
        </div>
      )}

      {/* MODAL: PUSH ANNOTATIONS TO CVAT (OUTBOUND SYNC) */}
      {showCVATPushModal && (
        <div className="modal-backdrop">
          <section className="modal qc-modal">
            <div className="modal-heading">
              <div>
                <div className="eyebrow">GỬI NHÃN SANG CÔNG CỤ CHỈNH SỬA</div>
                <h2>Gửi nhãn sang CVAT</h2>
              </div>
              <button className="icon-button" aria-label="Đóng cửa sổ gửi nhãn sang CVAT" onClick={() => setShowCVATPushModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Chọn nguồn nhãn và cách gửi bên dưới. Các khung bao được chuyển sang CVAT để bạn tiếp tục kiểm tra và chỉnh sửa. Nhãn AI gửi sang vẫn là gợi ý, chưa được coi là nhãn đã duyệt.
            </p>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                void handlePushToCVAT();
              }}
            >
              <div className="form-group">
                <label>
                  <span>Nguồn dữ liệu đẩy sang CVAT:</span>
                  <select
                    value={cvatPushSource}
                    onChange={(e) => setCvatPushSource(e.target.value as "merged" | "predictions" | "annotations")}
                  >
                    <option value="merged">
                      Nhãn hiện có + vật thể thiếu đã được chấp nhận
                    </option>
                    <option value="predictions">
                      Nhãn gợi ý từ AI · Cần người kiểm tra lại
                    </option>
                    <option value="annotations">
                      Chỉ nhãn hiện có trong AnnoPilot
                    </option>
                  </select>
                  <small className="muted">
                    {cvatPushSource === "merged" && "Giữ khung bao hiện có trong AnnoPilot và bổ sung các vật thể thiếu đã được chấp nhận hoặc ghi nhận xử lý."}
                    {cvatPushSource === "predictions" && "Gửi dự đoán AI làm nhãn gợi ý để bạn kiểm tra và sửa trên CVAT."}
                    {cvatPushSource === "annotations" && "Chỉ gửi các khung bao hiện có trong AnnoPilot, không tự thêm dự đoán AI."}
                  </small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>Chế độ đẩy nhãn:</span>
                  <select
                    value={cvatPushMode}
                    onChange={(e) => setCvatPushMode(e.target.value as "replace" | "create")}
                  >
                    <option value="create">
                      Thêm khung bao mới · Giữ nguyên khung bao cũ
                    </option>
                    <option value="replace">
                      Thay thế khung bao 2D của các ảnh trong bộ dữ liệu
                    </option>
                  </select>
                  <small className="muted">
                    {cvatPushMode === "create"
                      ? "Thêm khung bao chưa trùng chính xác với khung cũ. Cách này không sửa loại của khung cũ; hãy sửa trên CVAT rồi đồng bộ lại."
                      : "Thay các khung bao 2D trên ảnh thuộc bộ dữ liệu; giữ ảnh khác, dạng nhãn khác và thẻ. Hãy đồng bộ nhãn mới nhất về trước để tránh mất thay đổi trên CVAT. Chưa hỗ trợ gửi lại đường bám vật thể trong video."}
                  </small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>Mã truy cập CVAT (PAT):</span>
                  <input
                    type="password"
                    value={cvatToken}
                    onChange={(e) => setCvatToken(e.target.value)}
                    required
                  />
                  <small className="muted">Dùng mã truy cập do CVAT cấp để cho phép ứng dụng gửi nhãn.</small>
                </label>
              </div>

              <div className="qc-reason-box" style={{ margin: "1rem 0" }}>
                <Database size={16} />
                <span>
                  Đích đến: <strong>{cvatBaseUrl}</strong> · Task ID:{" "}
                  <strong>{cvatTaskId || dataset.cvat_task_id || "Chưa cấu hình"}</strong>
                  {cvatJobId || dataset.cvat_job_id ? ` · Job ID: ${cvatJobId || dataset.cvat_job_id}` : ""}
                </span>
              </div>

              <div className="modal-actions">
                <button type="button" className="secondary" onClick={() => setShowCVATPushModal(false)}>
                  Hủy bỏ
                </button>
                <button type="submit" className="primary btn-cvat-primary">
                  <Send size={16} /> Gửi nhãn sang CVAT
                </button>
              </div>
            </form>
          </section>
        </div>
      )}

      {/* Multi-Model Ensemble (WBF) Modal */}
      {showEnsembleModal && (
        <div className="modal-backdrop" onClick={() => setShowEnsembleModal(false)}>
          <section
            className="modal-card card"
            style={{ maxWidth: "780px", width: "95%" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <div style={{ width: "36px", height: "36px", borderRadius: "10px", background: "rgba(168, 85, 247, 0.15)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Layers size={22} style={{ color: "#c084fc" }} />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: "1.2rem", fontWeight: 700 }}>
                    Tạo nhãn gợi ý bằng nhiều mô hình (WBF)
                  </h3>
                  <p className="muted" style={{ margin: 0, fontSize: "0.85rem" }}>
                    Nhiều mô hình cùng nhận diện ảnh. WBF gộp các khung bao gần trùng nhau và chia kết quả theo mức đồng thuận để bạn kiểm tra.
                  </p>
                </div>
              </div>
              <button className="icon-button" aria-label="Đóng cửa sổ tạo nhãn gợi ý" onClick={() => setShowEnsembleModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="beginner-callout">Nếu mới bắt đầu, giữ nguyên cài đặt và thử với 10 ảnh. Bấm <strong>Nhận diện và tạo nhãn gợi ý</strong> để chạy cả ba mô hình; kết quả cần được người kiểm tra xác nhận trước khi dùng làm nhãn chính thức.</p>

            {/* Architecture Explanation Banner */}
            <div style={{ background: "rgba(168, 85, 247, 0.08)", border: "1px solid rgba(168, 85, 247, 0.25)", borderRadius: "10px", padding: "1rem", margin: "1rem 0" }}>
              <div style={{ fontWeight: 600, color: "#d8b4fe", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Sparkles size={16} /> Hiểu ba mức kết quả
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", fontSize: "0.82rem" }}>
                <div style={{ background: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#86efac", marginBottom: "4px" }}>Mức 1 · Đồng thuận cao</div>
                  <div style={{ color: "#dcfce7", lineHeight: "1.3" }}>Nhiều mô hình cùng nhận diện một loại ở vị trí gần trùng nhau → <strong>Có thể dùng làm nhãn gợi ý sau khi kiểm tra</strong>.</div>
                </div>
                <div style={{ background: "rgba(245, 158, 11, 0.1)", border: "1px solid rgba(245, 158, 11, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#fcd34d", marginBottom: "4px" }}>Mức 2 · Khác loại vật thể</div>
                  <div style={{ color: "#fef3c7", lineHeight: "1.3" }}>Cùng vị trí nhưng khác loại, ví dụ ô tô và xe tải → <strong>Cần bạn chọn loại đúng</strong>.</div>
                </div>
                <div style={{ background: "rgba(56, 189, 248, 0.1)", border: "1px solid rgba(56, 189, 248, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#7dd3fc", marginBottom: "4px" }}>Mức 3 · Chưa đủ đồng thuận</div>
                  <div style={{ color: "#e0f2fe", lineHeight: "1.3" }}>Chỉ một mô hình phát hiện hoặc kết quả chưa đạt ngưỡng → <strong>Kiểm tra bằng chứng</strong>.</div>
                </div>
              </div>
            </div>

            {/* Option B Triad Preset Card */}
            <div
              style={{
                background: "linear-gradient(135deg, rgba(168, 85, 247, 0.12), rgba(59, 130, 246, 0.12))",
                border: "1px solid rgba(168, 85, 247, 0.35)",
                borderRadius: "10px",
                padding: "1rem",
                marginBottom: "1.25rem",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <Sparkles size={18} style={{ color: "#c084fc" }} />
                  <span style={{ fontWeight: 700, fontSize: "0.95rem", color: "#f3e8ff" }}>
                    Bắt đầu: cho 3 mô hình nhận diện ảnh
                  </span>
                  <span className="badge" style={{ background: "#7c3aed", color: "#fff", fontSize: "0.72rem", padding: "2px 8px" }}>
                    Khuyên dùng
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <select
                    aria-label="Số ảnh cần tạo nhãn gợi ý"
                    value={triadLimit === null ? "all" : String(triadLimit)}
                    onChange={(e) => setTriadLimit(e.target.value === "all" ? null : parseInt(e.target.value, 10))}
                    style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem", background: "rgba(0,0,0,0.4)", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.2)" }}
                    disabled={runningTriad}
                  >
                    <option value="all">Toàn bộ bộ dữ liệu</option>
                    <option value="10">Thử trước với 10 ảnh</option>
                    <option value="30">Thử trước với 30 ảnh</option>
                  </select>
                  <button
                    type="button"
                    className="primary"
                    style={{ padding: "0.35rem 0.85rem", fontSize: "0.82rem", background: "#7c3aed", display: "flex", alignItems: "center", gap: "0.4rem" }}
                    onClick={() => handleRunTriadEnsemble()}
                    disabled={busy || runningTriad}
                  >
                    {runningTriad ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}
                    <span>{runningTriad ? "Đang nhận diện và hợp nhất…" : "Nhận diện và tạo nhãn gợi ý"}</span>
                  </button>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.6rem", fontSize: "0.8rem" }}>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(168, 85, 247, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#e9d5ff", display: "flex", alignItems: "center", gap: "4px" }}>
                    ⚡ M1: {selectedYoloModel.toUpperCase()}
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>Nhận diện nhanh · Trọng số 1,0</div>
                </div>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(59, 130, 246, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#93c5fd", display: "flex", alignItems: "center", gap: "4px" }}>
                    🎯 M2: Faster R-CNN
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>Nhận diện hai giai đoạn · Trọng số 1,0</div>
                </div>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(245, 158, 11, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#fcd34d", display: "flex", alignItems: "center", gap: "4px" }}>
                    👁️ M3: RT-DETR-l
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>Nhận diện toàn ảnh · Trọng số 1,2</div>
                </div>
              </div>
            </div>

            {/* Model Selection & Weights */}
            <details className="advanced-options">
              <summary>Tùy chọn nâng cao · Nguồn dự đoán, trọng số và ngưỡng hợp nhất</summary>
              <p className="muted">Chỉ cần đổi khi bạn muốn hợp nhất lại dự đoán đã có. Trọng số càng lớn, mô hình càng có ảnh hưởng tới kết quả. Nút này không chạy nhận diện ảnh mới.</p>
            <div style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontWeight: 600, fontSize: "0.95rem", marginBottom: "0.5rem" }}>
                Nguồn dự đoán để hợp nhất · Chọn ít nhất 2 mô hình
              </div>
              {availableModelsData && availableModelsData.models.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {availableModelsData.models
                    .filter((m) => !m.startsWith("ensemble_") && !m.toLowerCase().startsWith('yolov8'))
                    .map((modelName) => {
                      const isSelected = selectedEnsembleModels.includes(modelName);
                      const count = availableModelsData.counts[modelName] || 0;
                      const weight = modelWeights[modelName] ?? 1.0;
                      return (
                        <div
                          key={modelName}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            padding: "0.6rem 0.9rem",
                            borderRadius: "8px",
                            background: isSelected ? "rgba(168, 85, 247, 0.12)" : "rgba(255, 255, 255, 0.03)",
                            border: `1px solid ${isSelected ? "rgba(168, 85, 247, 0.4)" : "rgba(255, 255, 255, 0.1)"}`,
                          }}
                        >
                          <label style={{ display: "flex", alignItems: "center", gap: "0.6rem", cursor: "pointer", margin: 0 }}>
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  setSelectedEnsembleModels([...selectedEnsembleModels, modelName]);
                                } else {
                                  setSelectedEnsembleModels(selectedEnsembleModels.filter((m) => m !== modelName));
                                }
                              }}
                            />
                            <span style={{ fontWeight: 600 }}>{modelName}</span>
                            {modelName.toLowerCase().includes("yolo") && (
                              <span className="badge" style={{ background: "rgba(168, 85, 247, 0.2)", color: "#d8b4fe", fontSize: "0.7rem", border: "1px solid rgba(168, 85, 247, 0.4)" }}>
                                CNN một giai đoạn
                              </span>
                            )}
                            {(modelName.toLowerCase().includes("faster") || modelName.toLowerCase().includes("rcnn")) && (
                              <span className="badge" style={{ background: "rgba(59, 130, 246, 0.2)", color: "#93c5fd", fontSize: "0.7rem", border: "1px solid rgba(59, 130, 246, 0.4)" }}>
                                CNN hai giai đoạn
                              </span>
                            )}
                            {(modelName.toLowerCase().includes("detr") || modelName.toLowerCase().includes("transformer")) && (
                              <span className="badge" style={{ background: "rgba(245, 158, 11, 0.2)", color: "#fcd34d", fontSize: "0.7rem", border: "1px solid rgba(245, 158, 11, 0.4)" }}>
                                Kiến trúc Transformer
                              </span>
                            )}
                            <span className="badge" style={{ background: "rgba(255, 255, 255, 0.1)", fontSize: "0.75rem" }}>
                              {count.toLocaleString()} dự đoán
                            </span>
                          </label>

                          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Trọng số:</span>
                            <input
                              type="number"
                              min="0.1"
                              max="5.0"
                              step="0.1"
                              value={weight}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value) || 1.0;
                                setModelWeights({ ...modelWeights, [modelName]: val });
                              }}
                              style={{ width: "70px", padding: "0.25rem 0.5rem", fontSize: "0.85rem", textAlign: "center" }}
                              disabled={!isSelected}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              ) : (
                <div className="alert warning" style={{ fontSize: "0.85rem", padding: "0.75rem" }}>
                  Chưa có dự đoán. Hãy bấm Nhận diện và tạo nhãn gợi ý ở trên, hoặc nhập tệp dự đoán bên dưới.
                </div>
              )}
            </div>

            {/* WBF Hyperparameters */}
            <div style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontWeight: 600, fontSize: "0.95rem", marginBottom: "0.5rem" }}>
                Tùy chỉnh cách hợp nhất khung bao
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>Mức trùng khớp vị trí (IoU):</span>
                    <input
                      type="number"
                      min="0.1"
                      max="0.95"
                      step="0.05"
                      value={ensembleIouThresh}
                      onChange={(e) => setEnsembleIouThresh(parseFloat(e.target.value) || 0.55)}
                    />
                    <small className="muted">Hai khung trùng nhau đủ mức này sẽ được xem là cùng vật thể. Mặc định: 0,55.</small>
                  </label>
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>Độ tin cậy tối thiểu của kết quả:</span>
                    <input
                      type="number"
                      min="0.05"
                      max="1.0"
                      step="0.05"
                      value={ensembleConfThresh}
                      onChange={(e) => setEnsembleConfThresh(parseFloat(e.target.value) || 0.30)}
                    />
                    <small className="muted">Bỏ kết quả có độ tin cậy thấp hơn mức này. Mặc định: 0,30.</small>
                  </label>
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>Ngưỡng đồng thuận cao (mức 1):</span>
                    <input
                      type="number"
                      min="0.5"
                      max="1.0"
                      step="0.05"
                      value={ensembleHighConsensusConf}
                      onChange={(e) => setEnsembleHighConsensusConf(parseFloat(e.target.value) || 0.85)}
                    />
                    <small className="muted">Ngưỡng để xếp mức đồng thuận cao. Đây không phải xác suất nhãn đúng. Mặc định: 0,85.</small>
                  </label>
                </div>
              </div>
            </div>

            {/* Upload Additional Model File */}
            <details style={{ marginBottom: "1.25rem", background: "rgba(255, 255, 255, 0.02)", border: "1px solid rgba(255, 255, 255, 0.08)", borderRadius: "8px", padding: "0.75rem" }}>
              <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.9rem", color: "#c084fc" }}>
                + Nạp thêm dự đoán từ mô hình khác (RT-DETR, Faster R-CNN, v.v.)
              </summary>
              <form onSubmit={handleUploadAdditionalModel} style={{ marginTop: "0.75rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                  <div className="form-group" style={{ margin: 0 }}>
                    <label style={{ fontSize: "0.8rem" }}>
                      <span>Tên định danh mô hình:</span>
                      <input
                        type="text"
                        placeholder="Ví dụ: mo_hinh_giao_thong_1"
                        value={newModelName}
                        onChange={(e) => setNewModelName(e.target.value)}
                        required
                      />
                    </label>
                  </div>
                  <div className="form-group" style={{ margin: 0 }}>
                    <label style={{ fontSize: "0.8rem" }}>
                      <span>Chọn file JSON dự đoán:</span>
                      <input
                        type="file"
                        accept=".json"
                        onChange={(e) => setNewModelFile(e.target.files?.[0] || null)}
                        required
                      />
                    </label>
                  </div>
                </div>
                <button type="submit" className="secondary" style={{ alignSelf: "flex-end" }} disabled={busy || !newModelFile}>
                  <Upload size={14} /> Nhập dự đoán và hợp nhất
                </button>
              </form>
            </details>
            </details>

            <details className="advanced-options">
              <summary>Thử nghiệm nâng cao · So sánh nhiều lượt nhận diện có nhiễu</summary>
              <WBFExperimentPanel key={dataset.id} datasetId={dataset.id} yoloModel={selectedYoloModel} />
            </details>

            {/* Last Execution Results */}
            {ensembleResult && (
              <div style={{ background: "rgba(168, 85, 247, 0.1)", border: "1px solid rgba(168, 85, 247, 0.3)", borderRadius: "8px", padding: "1rem", marginBottom: "1.25rem" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                  <span style={{ fontWeight: 700, color: "#d8b4fe" }}>
                    Kết quả hợp nhất gần nhất ({ensembleResult.total_fused_predictions?.toLocaleString()} khung bao)
                  </span>
                  <button
                    className="primary"
                    style={{ padding: "0.35rem 0.75rem", fontSize: "0.8rem", background: "#7c3aed" }}
                    onClick={() => {
                      setShowEnsembleModal(false);
                      setCvatPushSource("predictions");
                      setShowCVATPushModal(true);
                    }}
                  >
                    <Send size={14} /> Gửi nhãn gợi ý sang CVAT
                  </button>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", textAlign: "center" }}>
                  <div style={{ background: "rgba(34, 197, 94, 0.15)", borderRadius: "8px", padding: "0.6rem" }}>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#86efac" }}>
                      {ensembleResult.tier_breakdown?.tier_1_high_consensus || 0}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#dcfce7" }}>
                      Mức 1 · Đồng thuận cao ·{" "}
                      {ensembleResult.total_fused_predictions
                        ? Math.round(((ensembleResult.tier_breakdown?.tier_1_high_consensus || 0) / ensembleResult.total_fused_predictions) * 100)
                        : 0}
                      %
                    </div>
                  </div>
                  <div style={{ background: "rgba(245, 158, 11, 0.15)", borderRadius: "8px", padding: "0.6rem" }}>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#fcd34d" }}>
                      {ensembleResult.tier_breakdown?.tier_2_class_disagreement || 0}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#fef3c7" }}>Mức 2 · Khác loại vật thể</div>
                  </div>
                  <div style={{ background: "rgba(56, 189, 248, 0.15)", borderRadius: "8px", padding: "0.6rem" }}>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#7dd3fc" }}>
                      {ensembleResult.tier_breakdown?.tier_3_model_divergence || 0}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#e0f2fe" }}>Mức 3 · Chưa đủ đồng thuận</div>
                  </div>
                </div>
              </div>
            )}

            <div className="modal-actions" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <button type="button" className="secondary" onClick={() => setShowEnsembleModal(false)}>
                Đóng
              </button>
              <button
                type="button"
                className="primary"
                style={{ background: "#9333ea", borderColor: "#a855f7" }}
                disabled={busy || selectedEnsembleModels.length < 2}
                title={selectedEnsembleModels.length < 2 ? "Cần ít nhất 2 nguồn dự đoán. Dùng Nhận diện và tạo nhãn gợi ý nếu chưa có dự đoán." : "Chỉ hợp nhất lại các dự đoán đã lưu, không chạy nhận diện ảnh mới."}
                onClick={handleRunEnsemble}
              >
                <Sparkles size={16} /> Hợp nhất dự đoán đã có
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

