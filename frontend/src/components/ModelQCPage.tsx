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
import { QCEvaluationPanel } from './QCEvaluationPanel';
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
      title="Bấm để cuộn lên và phóng to (zoom) vào vật thể này trên ảnh chính"
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
          <Crosshair size={11} /> Định vị & Zoom
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
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

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
  const [triadLimit, setTriadLimit] = useState<number | null>(null);
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
          api<QCAuditReport>(`/datasets/${dataset.id}/qc/report`),
          api<Sample[]>(`/datasets/${dataset.id}/qc/queue?include_clean=true`),
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
    try {
      setLoading(true);
      setError("");
      // Fetch report & queue in parallel
      const [reportData, queueData] = await Promise.all([
        api<QCAuditReport>(`/datasets/${dataset.id}/qc/report`),
        api<Sample[]>(`/datasets/${dataset.id}/qc/queue?include_clean=true`),
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
            setSelectedSample(detailed);
          } catch {
            setSelectedSample(matchingCurrent);
          }
        } else {
          const firstFlagged = queueData.find((s) => (s.qc_issue_count || 0) > 0) || queueData[0];
          setSelectedSample(firstFlagged);
        }
      } else {
        setSelectedSample(null);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load QC audit data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadQCData();
  }, [dataset.id]);

  // Refresh single sample detail
  const refreshCurrentSample = async (sampleId: number) => {
    try {
      const detailed = await api<Sample>(`/samples/${sampleId}`);
      setSelectedSample(detailed);
      // Update in queue list
      setQueue((prev) => prev.map((s) => (s.id === sampleId ? { ...s, ...detailed } : s)));
    } catch (e) {
      console.error(e);
    }
  };

  // Run Pretrained Model Detection
  const handleRunDetector = () => {
    void act("Running Pretrained Detector & Audit…", async () => {
      try {
        await api(`/datasets/${dataset.id}/qc/detect?model_name=${selectedYoloModel}`, { method: "POST" });
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Detector execution failed");
      }
    });
  };

  // Start Incremental Streaming Queue (P0)
  const handleStartQueue = async (forceRepredict = false) => {
    void act("Khởi động hàng đợi xử lý streaming từng frame…", async () => {
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
    void act("Đang kiểm tra chi tiết frame…", async () => {
      try {
        await api(`/datasets/${dataset.id}/qc/frame/${sampleId}/audit?force_repredict=true&model_name=${selectedYoloModel}`, { method: "POST" });
        await refreshCurrentSample(sampleId);
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Kiểm tra frame thất bại");
      } finally {
        setAuditingFrameId(null);
      }
    });
  };

  // Verify and sync single frame against CVAT
  const handleVerifyFrameCVAT = async (sampleId: number) => {
    setVerifyingFrameId(sampleId);
    void act("Đang đối soát frame với CVAT…", async () => {
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
        setError(err?.message || "Đối soát frame với CVAT thất bại");
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
    void act("Đang đồng bộ nhãn từ CVAT (Source of Truth)…", async () => {
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
    void act("Simulating QC Benchmark Injections…", async () => {
      try {
        await api(
          `/datasets/${dataset.id}/qc/synthetic-benchmark?missing_ratio=${benchmarkMissingRatio}&wrong_class_ratio=${benchmarkWrongRatio}&source_model=synthetic_oracle_v1`,
          { method: "POST" },
        );
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Synthetic benchmark failed");
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
    void act("Uploading predictions & auditing…", async () => {
      try {
        const response = await fetch(`/api/datasets/${dataset.id}/qc/predictions/upload`, {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.detail || "Upload failed");
        }
        await loadQCData();
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Predictions upload failed");
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
    if (selectedEnsembleModels.length === 0) {
      setError("Vui lòng chọn ít nhất 1 model để thực hiện Ensemble.");
      return;
    }
    void act("Đang kết hợp dự đoán (WBF Ensemble & Tri-Tier Classification)...", async () => {
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
        setError(err?.message || "Thực thi WBF Ensemble thất bại");
      }
    });
  };

  // Upload an additional model prediction file and fuse
  const handleUploadAdditionalModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newModelFile) {
      setError("Vui lòng chọn file JSON dự đoán của model");
      return;
    }
    const modelName = newModelName.trim() || `model_${Date.now()}`;
    void act(`Đang tải lên model '${modelName}' và kết hợp WBF...`, async () => {
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
        setError(err?.message || "Upload model và kết hợp thất bại");
      }
    });
  };

  // Run Option B Triad Ensemble (YOLO26n + Faster R-CNN + RT-DETR)
  const handleRunTriadEnsemble = async (customLimit?: number | null) => {
    setRunningTriad(true);
    const limitToUse = customLimit !== undefined ? customLimit : triadLimit;
    void act("Đang chạy inference Bộ 3 Mô hình Phương án B (YOLO26 + Faster R-CNN + RT-DETR)...", async () => {
      try {
        const res = await api<TriadRunResponse>(`/datasets/${dataset.id}/ensemble/run-triad`, {
          method: "POST",
          body: JSON.stringify({
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
        });
        if (res.ensemble_result) {
          setEnsembleResult(res.ensemble_result);
        }
        if (!res.success) setError(`Inference chưa đầy đủ: ${res.skipped_frames?.length || 0} frame thiếu media.`);
        const modelsData = await api<DatasetModelsResponse>(`/datasets/${dataset.id}/ensemble/models`);
        setAvailableModelsData(modelsData);
        setSelectedEnsembleModels((modelsData.active_models || modelsData.models).filter((m) => !m.startsWith("ensemble_") && !m.startsWith('synthetic')));
        await onRefreshDataset();
        await loadQCData();
        alert(
          `🎉 Hoàn tất Phương án B!\n` +
          `• Frames đã xử lý: ${res.frames_processed}\n` +
          `• Tổng số dự đoán: ${res.total_predictions.toLocaleString()} boxes\n` +
          `  - ${selectedYoloModel.toUpperCase()} (1-Stage): ${res.model_counts?.[selectedYoloModel] || 0}\n` +
          `  - Faster R-CNN (2-Stage): ${res.model_counts?.fasterrcnn_mobilenet_v3 || 0}\n` +
          `  - RT-DETR-l (Transformer): ${res.model_counts?.rtdetr_l || 0}\n` +
          `• Pre-labels đồng thuận WBF: ${res.ensemble_result?.total_fused_predictions || 0} boxes!`
        );
      } catch (err: any) {
        setError(err?.message || "Chạy inference Phương án B thất bại");
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
        const reportData = await api<QCAuditReport>(`/datasets/${dataset.id}/qc/report`);
        setReport(reportData);
      } catch (err: any) {
        setError(err?.message || "Failed to resolve QC issue");
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
          const reportData = await api<QCAuditReport>(`/datasets/${dataset.id}/qc/report`);
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
        const hasIssue = item.qc_issues?.some((i) => i.issue_type === issueTypeFilter);
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
      if (p.source_model) set.add(p.source_model);
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
    let preds = selectedSample.predictions.filter((p) => (p.confidence ?? 1.0) >= minPredConf);
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
            <span className="badge badge-accent">ĐỀ TÀI N2-04D</span>
            <span>QUALITY CONTROL CHECK · BDD100K TAXONOMY</span>
          </div>
          <h1>
            Model-Assisted QC: <span className="highlight">Bắt vật thể bị sót & sai class</span>
          </h1>
          <p className="muted">
            So sánh nhãn người gán với dự đoán của Pretrained Detector. Tự động phát hiện các vật thể bị bỏ sót
            (Missing Object) và gán sai nhãn (Wrong Class), đồng thời đẩy các ca đáng nghi nhất lên đầu danh sách review.
          </p>
        </div>

        <div className="qc-actions-bar">
          <label style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
            YOLO
            <select aria-label="Chọn biến thể YOLO26" value={selectedYoloModel}
              disabled={busy || isQueueRunning || runningTriad}
              onChange={e => selectYoloModel(e.target.value as YOLOVariant)}>
              <option value="yolo26n">YOLO26n · nano</option>
              <option value="yolo26s">YOLO26s · small</option>
            </select>
          </label>
          {/* Action 1: Chạy Streaming Queue (Primary) */}
          <button
            className="primary btn-streaming-start"
            disabled={busy || isQueueRunning}
            onClick={() => handleStartQueue(false)}
            title="Chạy kiểm tra lần lượt từng frame dạng streaming/incremental không block hệ thống"
          >
            <Radio size={16} /> {isQueueRunning ? "Đang xử lý Streaming..." : "Chạy Queue QC"}
          </button>

          {/* Action 2: Multi-Model Ensemble & Pre-labeling (WBF) */}
          <button
            className="secondary btn-ensemble-trigger"
            style={{ borderColor: "#a855f7", color: "#c084fc", background: "rgba(168, 85, 247, 0.08)" }}
            disabled={busy}
            onClick={() => void handleOpenEnsembleModal()}
            title="Kết hợp dự đoán đa mô hình (Multi-Model WBF) để tạo pre-label đồng thuận chất lượng cao"
          >
            <Layers size={16} /> Multi-Model WBF
          </button>

          {/* Action 3: Đồng bộ từ CVAT (Inbound) */}
          <button
            className="secondary btn-cvat-sync-trigger"
            disabled={busy}
            onClick={() => setShowCVATSyncModal(true)}
            title="Đồng bộ file nhãn mới nhất từ CVAT để đối soát và tự động gỡ cảnh báo"
          >
            <FileUp size={16} /> Đồng bộ từ CVAT
          </button>

          {/* Action 4: Đẩy nhãn sang CVAT (Outbound) */}
          <button
            className="secondary btn-cvat-push-trigger"
            disabled={busy}
            onClick={() => setShowCVATPushModal(true)}
            title="Đẩy các nhãn đã được duyệt hoặc nhãn AI sang CVAT Server qua REST API"
          >
            <Send size={16} /> Đẩy nhãn sang CVAT
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
                <div className="qc-dropdown-section-title">Dữ liệu & Model</div>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowUploadModal(true);
                  }}
                >
                  <Upload size={15} style={{ color: "var(--accent)" }} />
                  <span>Upload Predictions JSON</span>
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
                  <span>Inference Bộ 3 Phương Án B (Triad)</span>
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
                  <span>Chạy Detector Batch (Cũ)</span>
                </button>

                <div className="qc-dropdown-divider" />
                <div className="qc-dropdown-section-title">Thử nghiệm & Benchmark</div>
                <button
                  type="button"
                  className="qc-dropdown-item"
                  onClick={() => {
                    setShowMoreToolsMenu(false);
                    setShowBenchmarkModal(true);
                  }}
                >
                  <Sparkles size={15} style={{ color: "#facc15" }} />
                  <span>Simulate Benchmark Sandbox</span>
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
                  <span>Sơ đồ Kiến trúc CVAT × QC</span>
                </button>
              </div>
            )}
          </div>

          {/* Action 6: Nút Refresh icon */}
          <button className="icon-button" title="Làm mới dữ liệu QC" disabled={busy} onClick={() => void loadQCData()}>
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
                Multi-Model WBF: Đã kết hợp thành công {ensembleResult.total_fused_predictions} bounding boxes từ {ensembleResult.models_used?.join(", ")}
              </strong>
              <div style={{ fontSize: "0.85rem", color: "#e9d5ff", marginTop: "2px" }}>
                🎯 Tier 1 (Đồng thuận cao): <strong>{ensembleResult.tier_breakdown?.tier_1_high_consensus || 0}</strong> ({ensembleResult.total_fused_predictions ? Math.round(((ensembleResult.tier_breakdown?.tier_1_high_consensus || 0) / ensembleResult.total_fused_predictions) * 100) : 0}%) ·{" "}
                ⚠️ Tier 2 (Bất đồng class): <strong>{ensembleResult.tier_breakdown?.tier_2_class_disagreement || 0}</strong> ·{" "}
                🔍 Tier 3 (Chỉ 1 model): <strong>{ensembleResult.tier_breakdown?.tier_3_model_divergence || 0}</strong>
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
              <Send size={14} /> Đẩy Pre-label sang CVAT
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
                Đã đẩy thành công {cvatPushResult.pushed_shapes_count} bounding boxes sang CVAT Task #{cvatPushResult.cvat_task_id || dataset.cvat_task_id} (Job #{cvatPushResult.cvat_job_id || dataset.cvat_job_id}).
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
        <div className="alert error">
          <AlertTriangle size={18} />
          <span>{error}</span>
          <button className="icon-button" onClick={() => setError("")}>
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
              <span>Hàng đợi kiểm tra Streaming Frame-by-Frame</span>
              {queueStatus?.status === "RUNNING" && (
                <span className="qc-streaming-pill-running">
                  <LoaderCircle size={12} className="spin" /> Đang chạy
                </span>
              )}
              {queueStatus?.status === "COMPLETED" && (
                <span className="qc-streaming-pill-completed">
                  <CheckCircle2 size={12} /> Đã hoàn thành ({queueStatus.total_frames} frames)
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
              <span>Tiến độ: <strong>{queueStatus?.processed_frames || 0}</strong> / {queueStatus?.total_frames || 0} frames ({queueStatus?.progress_percent || 0}%)</span>
              {queueStatus?.current_frame_name && (
                <span style={{ marginLeft: "1rem" }}>
                  Đang xử lý: <code style={{ color: "#38bdf8" }}>{queueStatus.current_frame_name}</code>
                </span>
              )}
            </div>
            <div>
              <span>Phát hiện nghi vấn: <strong style={{ color: "#f87171" }}>{queueStatus?.flagged_frames_count || 0} frames</strong></span>
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

      <QCEvaluationPanel key={dataset.id} datasetId={dataset.id} />
      {report && ((report.unprocessed_samples_count || 0) > 0 || (report.inference_error_samples_count || 0) > 0) && (
        <p role="status" className="card text-warning" style={{ padding: '1rem' }}>
          Độ phủ chưa đầy đủ: {report.unprocessed_samples_count || 0} frame chưa inference/import prediction; {report.inference_error_samples_count || 0} frame có lỗi inference. Không dùng các frame này để kết luận chất lượng.
        </p>
      )}
      {/* KPI Metrics Dashboard Cards */}
      {report && (
        <div className="grid four qc-kpi-grid">
          {/* Workload Reduction */}
          <div className="card qc-kpi-card highlight-kpi">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Frame không có cảnh báo</span>
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
              <strong>{report.clean_samples_count}</strong> / {report.total_samples} frame không có cảnh báo. Chưa chứng minh nhãn đúng hoặc công review đã giảm.
            </p>
          </div>

          {/* Flagged Samples */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Cần Review (Flagged)</span>
              <ShieldAlert size={20} className="text-warning" />
            </div>
            <div className="qc-kpi-value text-warning">
              {report.flagged_samples_count}
              <small> / {report.total_samples}</small>
            </div>
            <div className="qc-pill-row">
              <span className="pill pill-high">HIGH: {report.severity_breakdown.HIGH || 0}</span>
              <span className="pill pill-med">MED: {report.severity_breakdown.MEDIUM || 0}</span>
              <span className="pill pill-low">LOW: {report.severity_breakdown.LOW || 0}</span>
            </div>
            <p className="qc-kpi-note">Được tự động ưu tiên xếp hạng theo QC Score.</p>
          </div>

          {/* Missing Objects */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Sót vật thể (Missing)</span>
              <ScanSearch size={20} className="text-accent" />
            </div>
            <div className="qc-kpi-value text-accent">
              {report.total_missing_candidates}
            </div>
            <p className="qc-kpi-note">Model thấy vật thể nhưng annotator chưa vẽ bounding box.</p>
          </div>

          {/* Wrong Classes */}
          <div className="card qc-kpi-card">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Sai Class (Wrong Class)</span>
              <AlertTriangle size={20} className="text-purple" />
            </div>
            <div className="qc-kpi-value text-purple">
              {report.total_wrong_class_candidates}
            </div>
            <p className="qc-kpi-note">Box trùng nhau nhưng class không khớp theo ma trận nhầm lẫn.</p>
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
            title={isSidebarOpen ? "Đóng sidebar hàng đợi" : "Mở sidebar hàng đợi"}
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
                <h3>Hàng đợi ưu tiên review</h3>
                <p className="muted small">Xếp theo độ nghi vấn (QC Score cao nhất ở trên cùng)</p>
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
                    {lvl}
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
                  Chỉ sai class
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
                  const missingCount = issues.filter((i) => i.issue_type === "MISSING_OBJECT").length;
                  const wrongCount = issues.filter((i) => i.issue_type === "WRONG_CLASS").length;
                  const resolvedCount = issues.filter((i) => i.status !== "PENDING").length;

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
                          Score: {(item.qc_score || 0).toFixed(3)}
                        </span>
                        <span className={`badge ${item.qc_severity ? item.qc_severity.toLowerCase() : "clean"}`}>
                          {item.qc_processing_state === 'ERROR' ? 'Lỗi inference' : item.qc_processing_state === 'NOT_RUN' ? 'Chưa inference' : item.qc_severity === 'CLEAN' ? 'Không có cảnh báo' : item.qc_severity}
                        </span>
                      </div>

                      <div className="qc-item-filename" title={item.file_name}>
                        {item.file_name}
                      </div>

                      <div className="qc-item-chips">
                        <span className="chip chip-frame">
                          Frame #{item.frame_number ?? (item.id - 1)}
                        </span>
                        <a
                          href={getSampleCVATUrl(item)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="queue-cvat-link"
                          title="Mở frame này trong CVAT"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink size={11} /> CVAT
                        </a>
                        {missingCount > 0 && (
                          <span className="chip chip-missing">+{missingCount} Sót</span>
                        )}
                        {wrongCount > 0 && (
                          <span className="chip chip-wrong">+{wrongCount} Sai Class</span>
                        )}
                        {issues.length > 0 && resolvedCount === issues.length && (
                          <span className="chip chip-resolved">
                            <Check size={10} /> Đã sửa hết
                          </span>
                        )}
                        {issues.length === 0 && (
                          <span className="chip chip-clean">
                            <ShieldCheck size={10} /> Không có lỗi
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
                      SAMPLE #{selectedSample.id} · FRAME #{selectedSample.frame_number ?? (selectedSample.id - 1)} · {selectedSample.task_type}
                    </div>
                    <h2>{selectedSample.file_name}</h2>
                  </div>
                  <a
                    href={getSampleCVATUrl(selectedSample)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-cvat-primary"
                    title="Mở trực tiếp frame này trong CVAT Annotation Editor để kiểm tra hoặc sửa nhãn"
                  >
                    <ExternalLink size={16} />
                    <span>Open in CVAT (Frame #{selectedSample.frame_number ?? (selectedSample.id - 1)})</span>
                  </a>
                  <button
                    type="button"
                    className="secondary"
                    style={{ padding: "0.45rem 0.85rem", fontSize: "0.85rem", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                    disabled={busy || auditingFrameId === selectedSample.id}
                    onClick={() => handleAuditSingleFrame(selectedSample.id)}
                    title="Chạy lại phát hiện và đối soát riêng frame này theo yêu cầu"
                  >
                    <RefreshCw size={14} className={auditingFrameId === selectedSample.id ? "spin" : ""} />
                    <span>Kiểm tra lại Frame</span>
                  </button>
                </div>

                {/* Layer Toggles & Smart Filters */}
                <div className="qc-layer-toggles" style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                  <span className="toggle-label" style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--muted)" }}>Lớp:</span>
                  <button
                    className={`toggle-btn ${showGT ? "active gt" : ""}`}
                    onClick={() => setShowGT(!showGT)}
                    title="Bật/Tắt nhãn người gán (Ground Truth)"
                  >
                    <span className="color-dot gt" /> Nhãn người ({selectedSample.annotations?.length || 0})
                  </button>
                  <button
                    className={`toggle-btn ${showPred ? "active pred" : ""}`}
                    onClick={() => setShowPred(!showPred)}
                    title="Bật/Tắt dự đoán của Pretrained Model"
                  >
                    <span className="color-dot pred" /> Model ({displayedPredictions.length})
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
                        title="Lọc nguồn model hiển thị trên ảnh (tránh chồng lấn nhiều model)"
                      >
                        {sampleModels.some((m) => m.startsWith("ensemble_")) && (
                          <option value={sampleModels.find((m) => m.startsWith("ensemble_"))}>
                            ⭐ Chỉ hiện Ensemble WBF (Đồng thuận)
                          </option>
                        )}
                        {sampleModels
                          .filter((m) => !m.startsWith("ensemble_"))
                          .map((m) => (
                            <option key={m} value={m}>
                              Model: {m}
                            </option>
                          ))}
                        <option value="all">Tất cả mô hình ({selectedSample.predictions?.length || 0} boxes - Debug)</option>
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
                      <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Min Conf:</span>
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
                        title="Lọc bỏ các dự đoán model có độ tin cậy thấp"
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
                      <Maximize2 size={12} /> Fit
                    </button>
                  </div>

                  <div className="toolbar-group zoom-status-hint">
                    {zoomScale > 1 ? (
                      <span className="pan-active-hint">
                        <Move size={12} /> Kéo chuột để di chuyển (Pan) · Cuộn chuột để zoom
                      </span>
                    ) : (
                      <span className="zoom-default-hint">
                        💡 Cuộn chuột trên ảnh để zoom · Bấm snapshot ở thẻ để tự zoom cận cảnh
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
                    <i className="legend-dot gt" /> Nhãn người gán (Ground Truth)
                  </span>
                  <span>
                    <i className="legend-dot pred" /> Model ({activeModel.startsWith("ensemble_") ? "⭐ Ensemble WBF" : activeModel})
                  </span>
                  <span>
                    <i className="legend-dot wrong-class" /> [⇄] Sai Class (Tím)
                  </span>
                  <span>
                    <i className="legend-dot missing" /> [!] Sót vật thể (Đỏ)
                  </span>
                  <span style={{ marginLeft: "auto", fontSize: "0.78rem", color: "var(--muted)" }}>
                    💡 Rê chuột vào box để xem chi tiết nhãn · Bấm pin để cuộn xuống xử lý
                  </span>
                </div>
              </div>

              {/* Closed-Loop Resolution Issues Cards */}
              <div className="qc-issues-section">
                <div className="qc-issues-heading">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                    <h3>
                      Bất đồng phát hiện trên frame này ({selectedSample.qc_issues?.length || 0} nghi vấn)
                    </h3>
                  </div>
                  <p className="muted small">
                    Mỗi nghi vấn được đánh số thứ tự (#1, #2, ...) tương ứng chính xác với box trên ảnh. Đối chiếu nhãn và bấm 1-Click để xử lý.
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
                            Có {tier1Issues.length} / {pendingIssues.length} nghi vấn đồng thuận cao (Tier 1)
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
                        title="Áp dụng các gợi ý Tier 1 sau khi reviewer đã kiểm tra"
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
                      <h4>Frame hoàn toàn đồng thuận (Clean)</h4>
                      <p className="muted">
                        Chưa có cảnh báo trên frame này. Kiểm tra trạng thái inference và kiểm toán độc lập trước khi kết luận chất lượng nhãn.
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
                                  <strong>SÓT VẬT THỂ (MISSING OBJECT)</strong>
                                </>
                              ) : (
                                <>
                                  <AlertTriangle size={16} />
                                  <strong>SAI PHÂN LOẠI (WRONG CLASS)</strong>
                                </>
                              )}
                              {g && (
                                <span className="qc-box-coords-chip" title="Tọa độ Bounding Box [x1, y1 · width × height]">
                                  Box [{Math.round(g.x1)}, {Math.round(g.y1)} · {Math.round(g.x2 - g.x1)}×{Math.round(g.y2 - g.y1)}px]
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
                                  ⭐ Đồng thuận {issue.evidence.num_models_agreed || 2}/3 Model
                                </span>
                              )}
                              {issue.evidence?.consensus_tier === 2 && (
                                <span
                                  className="badge"
                                  style={{ background: "rgba(245, 158, 11, 0.22)", color: "#fcd34d", border: "1px solid rgba(245, 158, 11, 0.55)", fontWeight: 700 }}
                                  title={issue.evidence.tier_reason || "Bất đồng class giữa các mô hình"}
                                >
                                  ⚠️ Bất đồng Class ({issue.evidence.num_models_agreed || 2} models)
                                </span>
                              )}
                              <span className="badge badge-accent">
                                QC Score: {issue.qc_score.toFixed(3)}
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
                              <span>vs</span>
                            </div>

                            <div className="cmp-col model">
                              <span className="cmp-label">Pretrained Model Đề Xuất</span>
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
                              <summary>Bằng chứng từng model · {issue.evidence.detection_count ?? issue.evidence.contributing_models?.length} model phát hiện</summary>
                              {issue.evidence.per_model_predictions.map((p: any, index: number) => (
                                <p key={index}><strong>{p.model_id}</strong>: {p.label} · confidence {(p.confidence * 100).toFixed(1)}% · box [{p.geometry.x1}, {p.geometry.y1}, {p.geometry.x2}, {p.geometry.y2}]</p>
                              ))}
                              {!!issue.evidence.absent_models?.length && <p>Không có detection tương ứng: {issue.evidence.absent_models.join(', ')}</p>}
                              <p className="muted">Confidence chưa calibration; đồng thuận không thay thế xác minh của reviewer.</p>
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
                                  title="Mở frame này trong CVAT Annotation Editor để sửa"
                                >
                                  <ExternalLink size={15} />
                                  Mở trên CVAT (Frame #{issue.frame_number ?? (selectedSample.frame_number ?? (selectedSample.id - 1))})
                                </a>
                                <button
                                  className="btn-verify-frame-cvat"
                                  disabled={busy || verifyingFrameId === selectedSample.id}
                                  onClick={() => handleVerifyFrameCVAT(selectedSample.id)}
                                  title="Kiểm tra xem nhãn/box trên CVAT đã được sửa chưa để tự động giải phóng cảnh báo"
                                >
                                  <RefreshCw size={14} className={verifyingFrameId === selectedSample.id ? "spin" : ""} />
                                  Kiểm tra & Đồng bộ từ CVAT
                                </button>
                                <button
                                  className="btn-resolve-cvat"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "RESOLVED", "Đã sửa trên CVAT", issue)}
                                  title="Đánh dấu đã hoàn thành việc sửa nhãn trực tiếp trên CVAT"
                                >
                                  <CheckCircle2 size={15} />
                                  Đã sửa trên CVAT (Resolved)
                                </button>
                                <button
                                  className="btn-accept"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "ACCEPTED", "1-Click tự động sửa", issue)}
                                  title="Tự động thêm box hoặc sửa label ngay lập tức cho demo nhanh"
                                >
                                  <Sparkles size={15} />
                                  1-Click Sửa tự động (Demo)
                                </button>
                                <button
                                  className="btn-reject"
                                  disabled={busy}
                                  onClick={() => handleResolveIssue(issue.id, "FALSE_POSITIVE", "Báo sai của model", issue)}
                                  title="Bác bỏ vì đây là báo sai (False Positive) của Pretrained Model"
                                >
                                  <X size={15} />
                                  Báo sai (False Positive)
                                </button>
                              </>
                            ) : isResolved ? (
                              <div className="resolution-status resolved">
                                <CheckCircle2 size={18} />
                                <span>
                                  Đã sửa trên CVAT: Annotator đã hoàn tất việc sửa nhãn trong CVAT Editor.
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
                                  Đã sửa tự động: Dữ liệu Ground Truth đã được cập nhật thành công (1-Click).
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
                                  Đã xác nhận False Positive: Giữ nguyên nhãn của người gán.
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
                <h2>Simulate QC Benchmark</h2>
              </div>
              <button className="icon-button" onClick={() => setShowBenchmarkModal(false)}>
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
                <h2>Upload Model Predictions JSON</h2>
              </div>
              <button className="icon-button" onClick={() => setShowUploadModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Nạp kết quả dự đoán từ YOLO26, Faster R-CNN hoặc bất kỳ detector nào theo định dạng JSON chuẩn BDD100K.
            </p>

            <form onSubmit={handleUploadPredictions}>
              <div className="form-group">
                <label>
                  Tên model nhận diện:
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
                  <strong>File dự đoán JSON</strong>
                  <span>Hỗ trợ JSON format (frames / predictions list)</span>
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
                <h2>Cấu hình kết nối CVAT Server</h2>
              </div>
              <button className="icon-button" onClick={() => setShowCVATModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Kết nối QC Dashboard với phiên bản CVAT của nhóm để kích hoạt cơ chế Deep-link
              <code> [Open in CVAT] </code> mở thẳng frame cần sửa trong Annotation Editor.
            </p>

            <form onSubmit={handleSaveCVATConfig}>
              <div className="form-group">
                <label>
                  <span>CVAT Base URL (Server):</span>
                  <input
                    type="url"
                    placeholder="http://localhost:8080"
                    value={cvatBaseUrl}
                    onChange={(e) => setCvatBaseUrl(e.target.value)}
                    required
                  />
                  <small className="muted">Mặc định: http://localhost:8080 (hoặc domain CVAT triển khai của nhóm)</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>CVAT Task ID:</span>
                  <input
                    type="number"
                    placeholder="VD: 1, 42"
                    value={cvatTaskId}
                    onChange={(e) => setCvatTaskId(e.target.value)}
                  />
                  <small className="muted">ID của Task trong CVAT tương ứng với dataset này.</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>CVAT Job ID (Tùy chọn):</span>
                  <input
                    type="number"
                    placeholder="VD: 1"
                    value={cvatJobId}
                    onChange={(e) => setCvatJobId(e.target.value)}
                  />
                  <small className="muted">Nếu có Job ID, đường link sẽ trỏ trực tiếp vào Job thay vì cấp Task.</small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>CVAT Personal Access Token (PAT):</span>
                  <input
                    type="password"
                    placeholder=""
                    value={cvatToken}
                    onChange={(e) => setCvatToken(e.target.value)}
                  />
                  <small className="muted">Token dùng để gọi CVAT REST API v2 đẩy/nhận nhãn tự động (Authorization: Bearer PAT).</small>
                </label>
              </div>

              <div className="qc-reason-box" style={{ margin: "1rem 0" }}>
                <ExternalLink size={16} />
                <span>
                  Định dạng Deep-link:{" "}
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
              <button className="icon-button" onClick={() => setShowArchModal(false)}>
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
                    <span>CVAT (System of Record)</span>
                  </div>
                  <span className="arch-tier-tag">Nơi sửa dữ liệu</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Raw Video & Frames</strong>
                    <p>Lưu trữ hình ảnh gốc theo Frame Index (0, 1, 2... N)</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Annotation Editor</strong>
                    <p>Giao diện gán nhãn, điều chỉnh bounding box & class</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>CVAT REST API / Export</strong>
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
                    <span>Model QC Backend & Engine</span>
                  </div>
                  <span className="arch-tier-tag">Tầng đối soát tự động</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Pretrained Detector</strong>
                    <p>Inference dự đoán vị trí & nhãn vật thể (BDD100K classes)</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Heuristic Discrepancy Engine</strong>
                    <p>So khớp IoU, phát hiện Sót vật thể & Sai class</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Prioritization & Health Score</strong>
                    <p>Tính QC Risk Score, đẩy ca bất đồng lên đầu hàng đợi</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Deep-link Generator</strong>
                    <p>Sinh link trực tiếp tới frame: <code>/tasks/ID/jobs/ID?frame=N</code></p>
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
                    <span>QC Dashboard (Reviewer Workspace)</span>
                  </div>
                  <span className="arch-tier-tag">Tầng ra quyết định & Review</span>
                </div>
                <div className="arch-tier-grid">
                  <div className="arch-sub-box">
                    <strong>Overview & Workload Reduction</strong>
                    <p>Thống kê frame không có cảnh báo; chất lượng cần kiểm chứng độc lập</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Prioritized Review Queue</strong>
                    <p>Xếp hạng danh sách ảnh từ nguy cơ cao nhất xuống thấp</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Dual-Layer Visual Diff</strong>
                    <p>Bật/tắt nhãn người gán, model phát hiện, vùng nghi vấn</p>
                  </div>
                  <div className="arch-sub-box">
                    <strong>Closed-Loop Triage</strong>
                    <p>Đánh dấu <em>Resolved trên CVAT</em>, <em>1-Click Sửa tự động</em>, hoặc <em>False Positive</em></p>
                  </div>
                </div>
              </div>
            </div>

            <div className="arch-footer-callout">
              <strong>Nguyên tắc vận hành:</strong> QC Dashboard không thay thế CVAT mà đóng vai trò là tầng thông minh (Decision Layer). Reviewer dùng QC Dashboard để tìm ra các frame lỗi nhanh nhất, sau đó bấm <code>Open in CVAT</code> để chuyển thẳng sang CVAT sửa dữ liệu nguồn (System of Record).
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
                <h2>Đồng bộ nhãn từ CVAT (Source of Truth)</h2>
              </div>
              <button className="icon-button" onClick={() => setShowCVATSyncModal(false)}>
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
                  <span>Hoặc tải lên file Annotations thủ công (.xml hoặc .json — Tùy chọn):</span>
                  <input type="file" accept=".xml,.json" />
                </label>
                <small className="muted">
                  Nếu không chọn file, hệ thống sẽ tự động lấy nhãn trực tiếp từ CVAT server qua REST API theo PAT đã lưu.
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
                <div className="eyebrow">OUTBOUND SYNCHRONIZATION · ANNOTATION PUSH</div>
                <h2>Đẩy nhãn đã duyệt sang CVAT (REST API)</h2>
              </div>
              <button className="icon-button" onClick={() => setShowCVATPushModal(false)}>
                <X size={16} />
              </button>
            </div>

            <p className="muted">
              Hệ thống sẽ chuyển đổi toàn bộ bounding box hợp lệ trong AnnoPilot (bao gồm các nhãn đã được Human-in-the-loop duyệt / sửa) thành định dạng chuẩn của CVAT (rectangle) và đẩy trực tiếp vào CVAT Task/Job qua REST API.
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
                      🌟 Gộp thông minh (Khuyên dùng) — Giữ 100% nhãn người gán + Bổ sung vật thể AI tìm thấy (MISSING_OBJECT)
                    </option>
                    <option value="predictions">
                      🤖 Dự đoán của Model AI (Pre-labeling) — Đẩy toàn bộ vật thể AI phát hiện lên CVAT làm nhãn gợi ý
                    </option>
                    <option value="annotations">
                      👤 Nhãn người gán gốc (Ground Truth) — Chỉ đẩy các box người gán đã vẽ trong AnnoPilot
                    </option>
                  </select>
                  <small className="muted">
                    {cvatPushSource === "merged" && "Tự động gộp nhãn người gán và các vật thể AI phát hiện bị sót để annotator trên CVAT không phải vẽ lại từ đầu."}
                    {cvatPushSource === "predictions" && "Rất phù hợp cho quy trình Pre-labeling: đưa nhãn AI vào CVAT để người dùng chỉnh sửa nhanh."}
                    {cvatPushSource === "annotations" && "Chỉ xuất các bounding box người gán hiện có trong cơ sở dữ liệu AnnoPilot sang CVAT."}
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
                      🛡️ Bổ sung thêm (Append / Create — Mặc định an toàn, không làm mất nhãn trên CVAT)
                    </option>
                    <option value="replace">
                      ⚠️ Ghi đè toàn bộ (Replace — Xóa sạch các shape đang có trên CVAT Job và thay bằng AnnoPilot)
                    </option>
                  </select>
                  <small className="muted">
                    {cvatPushMode === "create"
                      ? "Chế độ này an toàn tuyệt đối: nhãn vẽ trực tiếp trên CVAT sẽ KHÔNG bị mất."
                      : "Cảnh báo: Nếu annotator vừa vẽ trực tiếp trên CVAT mà chưa Đồng bộ về AnnoPilot, các nhãn đó trên CVAT sẽ bị thay thế."}
                  </small>
                </label>
              </div>

              <div className="form-group">
                <label>
                  <span>CVAT Personal Access Token (PAT):</span>
                  <input
                    type="password"
                    value={cvatToken}
                    onChange={(e) => setCvatToken(e.target.value)}
                    required
                  />
                  <small className="muted">Token dùng để xác thực Bearer với CVAT API.</small>
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
                  <Send size={16} /> Bắt đầu đẩy nhãn sang CVAT
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
                    Multi-Model Ensemble: Pre-labeling (WBF)
                  </h3>
                  <p className="muted" style={{ margin: 0, fontSize: "0.85rem" }}>
                    Kết hợp dự đoán đa mô hình bằng Weighted Box Fusion & Phân tầng đồng thuận Tri-Tier
                  </p>
                </div>
              </div>
              <button className="icon-button" onClick={() => setShowEnsembleModal(false)}>
                <X size={16} />
              </button>
            </div>

            {/* Architecture Explanation Banner */}
            <div style={{ background: "rgba(168, 85, 247, 0.08)", border: "1px solid rgba(168, 85, 247, 0.25)", borderRadius: "10px", padding: "1rem", margin: "1rem 0" }}>
              <div style={{ fontWeight: 600, color: "#d8b4fe", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Sparkles size={16} /> Cơ chế đồng thuận 3 cấp độ (Tri-Tier Consensus Architecture)
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", fontSize: "0.82rem" }}>
                <div style={{ background: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#86efac", marginBottom: "4px" }}>🎯 Tier 1: Đồng thuận cao</div>
                  <div style={{ color: "#dcfce7", lineHeight: "1.3" }}>Các model cùng nhãn với IoU & conf cao → <strong>Đề xuất pre-label, cần xác minh</strong>.</div>
                </div>
                <div style={{ background: "rgba(245, 158, 11, 0.1)", border: "1px solid rgba(245, 158, 11, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#fcd34d", marginBottom: "4px" }}>⚠️ Tier 2: Bất đồng nhãn</div>
                  <div style={{ color: "#fef3c7", lineHeight: "1.3" }}>Model trùng vị trí nhưng lệch class (e.g. car vs truck) → <strong>Gắn cờ 1-Click review</strong>.</div>
                </div>
                <div style={{ background: "rgba(56, 189, 248, 0.1)", border: "1px solid rgba(56, 189, 248, 0.3)", borderRadius: "8px", padding: "0.6rem" }}>
                  <div style={{ fontWeight: 700, color: "#7dd3fc", marginBottom: "4px" }}>🔍 Tier 3: Lệch model</div>
                  <div style={{ color: "#e0f2fe", lineHeight: "1.3" }}>Một model phát hiện hoặc đồng thuận chưa đạt ngưỡng → <strong>Kiểm tra bằng chứng</strong>.</div>
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
                    Phương án B: Bộ 3 Mô hình Đa Kiến Trúc (Triad Ensemble)
                  </span>
                  <span className="badge" style={{ background: "#7c3aed", color: "#fff", fontSize: "0.72rem", padding: "2px 8px" }}>
                    Khuyên dùng
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <select
                    value={triadLimit === null ? "all" : String(triadLimit)}
                    onChange={(e) => setTriadLimit(e.target.value === "all" ? null : parseInt(e.target.value, 10))}
                    style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem", background: "rgba(0,0,0,0.4)", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.2)" }}
                    disabled={runningTriad}
                  >
                    <option value="all">Toàn bộ Dataset</option>
                    <option value="10">Thử nghiệm 10 frames</option>
                    <option value="30">Thử nghiệm 30 frames</option>
                  </select>
                  <button
                    type="button"
                    className="primary"
                    style={{ padding: "0.35rem 0.85rem", fontSize: "0.82rem", background: "#7c3aed", display: "flex", alignItems: "center", gap: "0.4rem" }}
                    onClick={() => handleRunTriadEnsemble()}
                    disabled={busy || runningTriad}
                  >
                    {runningTriad ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}
                    <span>{runningTriad ? "Đang suy luận..." : "⚡ Chạy Inference & WBF Phương Án B"}</span>
                  </button>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.6rem", fontSize: "0.8rem" }}>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(168, 85, 247, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#e9d5ff", display: "flex", alignItems: "center", gap: "4px" }}>
                    ⚡ M1: {selectedYoloModel.toUpperCase()}
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>CNN 1-Stage · YOLO26 · Trọng số 1.0</div>
                </div>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(59, 130, 246, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#93c5fd", display: "flex", alignItems: "center", gap: "4px" }}>
                    🎯 M2: Faster R-CNN
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>CNN 2-Stage · RPN + RoIAlign · Trọng số 1.0</div>
                </div>
                <div style={{ background: "rgba(0, 0, 0, 0.25)", borderRadius: "6px", padding: "0.55rem 0.75rem", border: "1px solid rgba(245, 158, 11, 0.2)" }}>
                  <div style={{ fontWeight: 600, color: "#fcd34d", display: "flex", alignItems: "center", gap: "4px" }}>
                    👁️ M3: RT-DETR-l
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: "0.75rem", marginTop: "2px" }}>Transformer · Global Attention · Trọng số 1.2</div>
                </div>
              </div>
            </div>

            {/* Model Selection & Weights */}
            <div style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontWeight: 600, fontSize: "0.95rem", marginBottom: "0.5rem" }}>
                1. Chọn mô hình tham gia Ensemble & Thiết lập trọng số (Model Weights):
              </div>
              {availableModelsData && availableModelsData.models.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {availableModelsData.models
                    .filter((m) => !m.startsWith("ensemble_"))
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
                                CNN 1-Stage
                              </span>
                            )}
                            {(modelName.toLowerCase().includes("faster") || modelName.toLowerCase().includes("rcnn")) && (
                              <span className="badge" style={{ background: "rgba(59, 130, 246, 0.2)", color: "#93c5fd", fontSize: "0.7rem", border: "1px solid rgba(59, 130, 246, 0.4)" }}>
                                CNN 2-Stage
                              </span>
                            )}
                            {(modelName.toLowerCase().includes("detr") || modelName.toLowerCase().includes("transformer")) && (
                              <span className="badge" style={{ background: "rgba(245, 158, 11, 0.2)", color: "#fcd34d", fontSize: "0.7rem", border: "1px solid rgba(245, 158, 11, 0.4)" }}>
                                Vision Transformer
                              </span>
                            )}
                            <span className="badge" style={{ background: "rgba(255, 255, 255, 0.1)", fontSize: "0.75rem" }}>
                              {count.toLocaleString()} predictions
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
                  Chưa có nguồn dự đoán nào trong dataset. Hãy tải lên file predictions bên dưới!
                </div>
              )}
            </div>

            {/* WBF Hyperparameters */}
            <div style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontWeight: 600, fontSize: "0.95rem", marginBottom: "0.5rem" }}>
                2. Tham số thuật toán Weighted Box Fusion (WBF):
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>IoU Clustered Threshold:</span>
                    <input
                      type="number"
                      min="0.1"
                      max="0.95"
                      step="0.05"
                      value={ensembleIouThresh}
                      onChange={(e) => setEnsembleIouThresh(parseFloat(e.target.value) || 0.55)}
                    />
                    <small className="muted">Ngưỡng IoU coi 2 box là 1 vật thể (Mặc định 0.55)</small>
                  </label>
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>Min Fused Confidence:</span>
                    <input
                      type="number"
                      min="0.05"
                      max="1.0"
                      step="0.05"
                      value={ensembleConfThresh}
                      onChange={(e) => setEnsembleConfThresh(parseFloat(e.target.value) || 0.30)}
                    />
                    <small className="muted">Độ tin cậy tối thiểu để giữ box sau WBF (Mặc định 0.30)</small>
                  </label>
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label style={{ fontSize: "0.8rem" }}>
                    <span>Tier 1 Consensus Conf:</span>
                    <input
                      type="number"
                      min="0.5"
                      max="1.0"
                      step="0.05"
                      value={ensembleHighConsensusConf}
                      onChange={(e) => setEnsembleHighConsensusConf(parseFloat(e.target.value) || 0.85)}
                    />
                    <small className="muted">Ngưỡng confidence cho Tier 1; không phải xác suất nhãn đúng (mặc định 0.85)</small>
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
                      <span>Tên model (Model Identifier):</span>
                      <input
                        type="text"
                        placeholder="e.g. rt-detr-l hoặc yolov9c"
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
                  <Upload size={14} /> Nạp & Hợp nhất vào Ensemble
                </button>
              </form>
            </details>

            {/* Last Execution Results */}
            {ensembleResult && (
              <div style={{ background: "rgba(168, 85, 247, 0.1)", border: "1px solid rgba(168, 85, 247, 0.3)", borderRadius: "8px", padding: "1rem", marginBottom: "1.25rem" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                  <span style={{ fontWeight: 700, color: "#d8b4fe" }}>
                    Kết quả hợp nhất gần nhất ({ensembleResult.total_fused_predictions?.toLocaleString()} boxes)
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
                    <Send size={14} /> Đẩy Pre-label sang CVAT
                  </button>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", textAlign: "center" }}>
                  <div style={{ background: "rgba(34, 197, 94, 0.15)", borderRadius: "8px", padding: "0.6rem" }}>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#86efac" }}>
                      {ensembleResult.tier_breakdown?.tier_1_high_consensus || 0}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#dcfce7" }}>
                      🎯 Tier 1 (Đồng thuận cao) ·{" "}
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
                    <div style={{ fontSize: "0.75rem", color: "#fef3c7" }}>⚠️ Tier 2 (Bất đồng nhãn)</div>
                  </div>
                  <div style={{ background: "rgba(56, 189, 248, 0.15)", borderRadius: "8px", padding: "0.6rem" }}>
                    <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#7dd3fc" }}>
                      {ensembleResult.tier_breakdown?.tier_3_model_divergence || 0}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#e0f2fe" }}>🔍 Tier 3 (Chỉ 1 model)</div>
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
                disabled={busy || selectedEnsembleModels.length === 0}
                onClick={handleRunEnsemble}
              >
                <Sparkles size={16} /> Chạy WBF Ensemble
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

