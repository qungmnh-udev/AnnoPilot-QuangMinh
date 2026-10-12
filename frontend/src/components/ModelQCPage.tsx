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
} from "lucide-react";
import { api, json } from "../api";
import type {
  Dataset,
  Sample,
  QCAuditReport,
  QCIssue,
  Prediction,
  Annotation,
  FrameQueueStatusResponse,
  CVATSyncResponse,
  CVATVerifyFrameResponse,
  CVATPushResponse,
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

  // Modals
  const [showBenchmarkModal, setShowBenchmarkModal] = useState<boolean>(false);
  const [benchmarkMissingRatio, setBenchmarkMissingRatio] = useState<number>(0.15);
  const [benchmarkWrongRatio, setBenchmarkWrongRatio] = useState<number>(0.15);

  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);
  const [uploadSourceModel, setUploadSourceModel] = useState<string>("yolov8-bdd100k");

  // CVAT Integration & Architecture Modal states
  const [showArchModal, setShowArchModal] = useState<boolean>(false);
  const [showCVATModal, setShowCVATModal] = useState<boolean>(false);
  const [cvatBaseUrl, setCvatBaseUrl] = useState<string>(dataset.cvat_base_url || "http://localhost:8080");
  const [cvatTaskId, setCvatTaskId] = useState<string>(dataset.cvat_task_id ? String(dataset.cvat_task_id) : "");
  const [cvatJobId, setCvatJobId] = useState<string>(dataset.cvat_job_id ? String(dataset.cvat_job_id) : "");
  const [cvatToken, setCvatToken] = useState<string>(dataset.cvat_token || "ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO");

  useEffect(() => {
    setCvatBaseUrl(dataset.cvat_base_url || "http://localhost:8080");
    setCvatTaskId(dataset.cvat_task_id ? String(dataset.cvat_task_id) : "");
    setCvatJobId(dataset.cvat_job_id ? String(dataset.cvat_job_id) : "");
    setCvatToken(dataset.cvat_token || "ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO");
  }, [dataset]);

  // Incremental Streaming Queue & CVAT Two-Way Sync states (P0)
  const [queueStatus, setQueueStatus] = useState<FrameQueueStatusResponse | null>(null);
  const [isQueueRunning, setIsQueueRunning] = useState<boolean>(false);
  const [showCVATSyncModal, setShowCVATSyncModal] = useState<boolean>(false);
  const [cvatSyncResult, setCvatSyncResult] = useState<CVATSyncResponse | null>(null);
  const [showCVATPushModal, setShowCVATPushModal] = useState<boolean>(false);
  const [cvatPushResult, setCvatPushResult] = useState<CVATPushResponse | null>(null);
  const [cvatPushMode, setCvatPushMode] = useState<"replace" | "create">("replace");
  const [verifyingFrameId, setVerifyingFrameId] = useState<number | null>(null);
  const [auditingFrameId, setAuditingFrameId] = useState<number | null>(null);

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

  // Sidebar collapse state
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);

  // Focused issue state for visual-diff linking
  const [focusedIssueId, setFocusedIssueId] = useState<number | null>(null);

  useEffect(() => {
    setFocusedIssueId(null);
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
        await api(`/datasets/${dataset.id}/qc/detect`, { method: "POST" });
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
        await api(`/datasets/${dataset.id}/qc/frame/${sampleId}/audit?force_repredict=true`, { method: "POST" });
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
          }),
        );
        setCvatPushResult(res);
        await onRefreshDataset();
      } catch (err: any) {
        setError(err?.message || "Đẩy nhãn sang CVAT thất bại");
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
          <button
            className="secondary btn-arch-highlight"
            disabled={busy}
            onClick={() => setShowArchModal(true)}
            title="Xem sơ đồ kiến trúc hệ thống CVAT × QC Dashboard theo tài liệu thiết kế"
          >
            <Workflow size={16} /> Kiến trúc CVAT × QC
          </button>
          <button
            className="secondary"
            disabled={busy}
            onClick={() => setShowCVATModal(true)}
            title="Cấu hình URL CVAT Server, Task ID và Job ID"
          >
            <Link2 size={16} /> Cấu hình CVAT {dataset.cvat_task_id ? `(#${dataset.cvat_task_id})` : ""}
          </button>
          <button
            className="primary btn-streaming-start"
            disabled={busy || isQueueRunning}
            onClick={() => handleStartQueue(false)}
            title="Chạy kiểm tra lần lượt từng frame dạng streaming/incremental không block hệ thống"
          >
            <Radio size={16} /> {isQueueRunning ? "Đang xử lý Streaming..." : "Chạy Queue Từng Frame"}
          </button>
          <button
            className="secondary btn-cvat-sync-trigger"
            disabled={busy}
            onClick={() => setShowCVATSyncModal(true)}
            title="Đồng bộ file nhãn mới nhất từ CVAT để đối soát và tự động gỡ cảnh báo"
          >
            <FileUp size={16} /> Đồng bộ từ CVAT
          </button>
          <button
            className="secondary btn-cvat-push-trigger"
            disabled={busy}
            onClick={() => setShowCVATPushModal(true)}
            title="Đẩy các nhãn đã được duyệt (Ground Truth chuẩn) sang CVAT Server qua REST API"
          >
            <Send size={16} /> Đẩy nhãn sang CVAT
          </button>
          {selectedSample && (
            <button
              className="secondary"
              disabled={busy || auditingFrameId === selectedSample.id}
              onClick={() => handleAuditSingleFrame(selectedSample.id)}
              title="Chạy lại phát hiện và đối soát riêng frame này theo yêu cầu"
            >
              <RefreshCw size={14} className={auditingFrameId === selectedSample.id ? "spin" : ""} />
              Kiểm tra Frame này
            </button>
          )}
          <button className="secondary" disabled={busy} onClick={handleRunDetector} title="Chạy detector hàng loạt trên toàn bộ dataset">
            <Bot size={16} /> Chạy toàn bộ (Batch)
          </button>
          <button className="secondary" disabled={busy} onClick={() => setShowBenchmarkModal(true)}>
            <Sparkles size={16} /> Simulate Benchmark
          </button>
          <button className="secondary" disabled={busy} onClick={() => setShowUploadModal(true)}>
            <Upload size={16} /> Upload Predictions JSON
          </button>
          <button className="icon-button" title="Refresh Audit" disabled={busy} onClick={() => void loadQCData()}>
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

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

      {/* KPI Metrics Dashboard Cards */}
      {report && (
        <div className="grid four qc-kpi-grid">
          {/* Workload Reduction */}
          <div className="card qc-kpi-card highlight-kpi">
            <div className="qc-kpi-top">
              <span className="qc-kpi-label">Workload Reduction</span>
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
              <strong>{report.clean_samples_count}</strong> / {report.total_samples} ảnh sạch được miễn kiểm tra thủ công.
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
                          {item.qc_severity || "CLEAN"}
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
                </div>

                {/* Layer Toggles */}
                <div className="qc-layer-toggles">
                  <span className="toggle-label">Hiển thị lớp:</span>
                  <button
                    className={`toggle-btn ${showGT ? "active gt" : ""}`}
                    onClick={() => setShowGT(!showGT)}
                    title="Bật/Tắt nhãn người gán (Ground Truth)"
                  >
                    <span className="color-dot gt" /> Nhãn người
                  </button>
                  <button
                    className={`toggle-btn ${showPred ? "active pred" : ""}`}
                    onClick={() => setShowPred(!showPred)}
                    title="Bật/Tắt dự đoán của Pretrained Model"
                  >
                    <span className="color-dot pred" /> Model dự đoán
                  </button>
                  <button
                    className={`toggle-btn ${showIssues ? "active issue" : ""}`}
                    onClick={() => setShowIssues(!showIssues)}
                    title="Bật/Tắt vùng nghi vấn lỗi"
                  >
                    <span className="color-dot issue" /> Vùng nghi vấn
                  </button>
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
                          return (
                            <g key={`gt-${ann.id}`} className="svg-box gt-box">
                              <rect
                                x={g.x1}
                                y={g.y1}
                                width={g.x2 - g.x1}
                                height={g.y2 - g.y1}
                              />
                              <text x={g.x1 + 4} y={Math.max(16, g.y1 - 4)}>
                                Người: {ann.label}
                              </text>
                            </g>
                          );
                        })}

                      {/* 2. Model Predictions (Dashed Orange/Amber) */}
                      {showPred &&
                        selectedSample.predictions?.map((pred) => {
                          const g = pred.geometry;
                          if (!g) return null;
                          return (
                            <g key={`pred-${pred.id}`} className="svg-box pred-box">
                              <rect
                                x={g.x1}
                                y={g.y1}
                                width={g.x2 - g.x1}
                                height={g.y2 - g.y1}
                              />
                              <text x={g.x1 + 4} y={Math.min((selectedSample.height || 720) - 6, g.y2 + 14)}>
                                Model: {pred.label} ({(pred.confidence * 100).toFixed(0)}%)
                              </text>
                            </g>
                          );
                        })}

                      {/* 3. Flagged Issue Highlights (Glowing Red / Magenta / Purple) */}
                      {showIssues &&
                        (() => {
                          const issues = selectedSample.qc_issues || [];
                          const indexed = issues.map((issue, idx) => ({ issue, idx }));
                          // Render focused issue last so it stays on top in SVG
                          const sorted = [...indexed].sort((a, b) => {
                            if (a.issue.id === focusedIssueId) return 1;
                            if (b.issue.id === focusedIssueId) return -1;
                            return 0;
                          });

                          return sorted.map(({ issue, idx }) => {
                            const g = issue.location;
                            if (!g) return null;
                            const isResolved = issue.status !== "PENDING";
                            const isFocused = focusedIssueId === issue.id;
                            const isDimmed = focusedIssueId !== null && !isFocused;
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

                            return (
                              <g
                                key={`issue-${issue.id}`}
                                className={`svg-box issue-highlight interactive ${isWrongClass ? "wrong-class" : "missing-object"
                                  } ${isResolved ? "resolved" : ""} ${isFocused ? "is-focused" : ""
                                  } ${isDimmed ? "is-dimmed" : ""}`}
                                onMouseEnter={() => setFocusedIssueId(issue.id)}
                                onMouseLeave={() => setFocusedIssueId(null)}
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

                                {/* Pill Badge Background */}
                                <rect
                                  className="issue-badge-bg"
                                  x={badgeX}
                                  y={badgeY - badgeH + 4}
                                  width={badgeW}
                                  height={badgeH}
                                  rx={4}
                                  ry={4}
                                />

                                {/* Pill Badge Text */}
                                <text
                                  className="issue-badge-text"
                                  x={badgeX + 8}
                                  y={badgeY}
                                >
                                  {labelText}
                                </text>

                                {/* Corner Number Pin Marker */}
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
                    <i className="legend-dot pred" /> Model phát hiện (Pretrained Detector)
                  </span>
                  <span>
                    <i className="legend-dot wrong-class" /> [⇄] Sai Class (Tím)
                  </span>
                  <span>
                    <i className="legend-dot missing" /> [!] Sót vật thể (Đỏ)
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
                        Dự đoán của model và nhãn người gán khớp chính xác với nhau. Frame này được tự động nghiệm thu mà không cần review lại.
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
              Nạp kết quả dự đoán từ YOLOv8, Faster R-CNN hoặc bất kỳ detector nào theo định dạng JSON chuẩn BDD100K.
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
                    placeholder="ISQjKy3C.KDdn6nz1T65dB0A0s1yhb2ZWIQVcSDAO"
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
                    <p>Đo lường % ảnh sạch được nghiệm thu tự động</p>
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
                  <span>Chế độ đẩy nhãn:</span>
                  <select
                    value={cvatPushMode}
                    onChange={(e) => setCvatPushMode(e.target.value as "replace" | "create")}
                  >
                    <option value="replace">Ghi đè nhãn (Replace shapes - Giữ nguyên video tracks nếu có)</option>
                    <option value="create">Bổ sung nhãn mới (Append / Create new shapes)</option>
                  </select>
                  <small className="muted">
                    Khuyến nghị <strong>Ghi đè nhãn (Replace shapes)</strong> để đồng bộ và làm sạch tuyệt đối với Ground Truth mới sau khi QC.
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
    </div>
  );
}
