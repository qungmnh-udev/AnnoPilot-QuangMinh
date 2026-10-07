import { useEffect, useState, useMemo } from "react";
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
} from "lucide-react";
import { api, json } from "../api";
import type { Dataset, Sample, QCAuditReport, QCIssue, Prediction, Annotation } from "../types";

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

  useEffect(() => {
    setCvatBaseUrl(dataset.cvat_base_url || "http://localhost:8080");
    setCvatTaskId(dataset.cvat_task_id ? String(dataset.cvat_task_id) : "");
    setCvatJobId(dataset.cvat_job_id ? String(dataset.cvat_job_id) : "");
  }, [dataset]);

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
          }),
        );
        await onRefreshDataset();
        await loadQCData();
      } catch (err: any) {
        setError(err?.message || "Lỗi lưu cấu hình CVAT");
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
          <button className="primary" disabled={busy} onClick={handleRunDetector}>
            <Bot size={16} /> Run Pretrained Detector
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

      {error && (
        <div className="alert error">
          <AlertTriangle size={18} />
          <span>{error}</span>
          <button className="icon-button" onClick={() => setError("")}>
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
      <div className="qc-main-grid">
        {/* LEFT COLUMN: PRIORITIZED QUEUE */}
        <section className="card qc-queue-card">
          <div className="qc-queue-header">
            <div>
              <h3>Hàng đợi ưu tiên review</h3>
              <p className="muted small">Xếp theo độ nghi vấn (QC Score cao nhất ở trên cùng)</p>
            </div>
            <span className="badge badge-primary">{filteredQueue.length} ảnh</span>
          </div>

          {/* Filter Bar */}
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

          {/* Queue List */}
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
                    className={`qc-queue-item ${isSelected ? "selected" : ""} ${
                      item.qc_severity ? item.qc_severity.toLowerCase() : ""
                    }`}
                    onClick={() => refreshCurrentSample(item.id)}
                  >
                    <div className="qc-item-top">
                      <span className="qc-item-rank">#{idx + 1}</span>
                      <span
                        className={`badge qc-score-badge ${
                          item.qc_severity ? item.qc_severity.toLowerCase() : ""
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

              {/* Visual Diff Canvas Stage */}
              <div className="qc-stage-container">
                <div
                  className="qc-canvas-stage"
                  style={{
                    aspectRatio: `${selectedSample.width || 1280}/${selectedSample.height || 720}`,
                  }}
                >
                  {selectedSample.media_url ? (
                    <img src={selectedSample.media_url} alt={selectedSample.file_name} />
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

                    {/* 3. Flagged Issue Highlights (Glowing Red / Magenta) */}
                    {showIssues &&
                      selectedSample.qc_issues?.map((issue) => {
                        const g = issue.location;
                        if (!g) return null;
                        const isResolved = issue.status !== "PENDING";
                        return (
                          <g
                            key={`issue-${issue.id}`}
                            className={`svg-box issue-highlight ${
                              isResolved ? "resolved" : ""
                            }`}
                          >
                            <rect
                              x={g.x1 - 3}
                              y={g.y1 - 3}
                              width={g.x2 - g.x1 + 6}
                              height={g.y2 - g.y1 + 6}
                            />
                            <text x={g.x1} y={Math.max(14, g.y1 - 10)}>
                              [!] {issue.issue_type === "MISSING_OBJECT" ? "Sót: " : "Đổi: "}
                              {issue.suggested_label}
                            </text>
                          </g>
                        );
                      })}
                  </svg>
                </div>

                <div className="qc-stage-legend">
                  <span>
                    <i className="legend-dot gt" /> Nhãn người gán (Ground Truth)
                  </span>
                  <span>
                    <i className="legend-dot pred" /> Model phát hiện (Pretrained Detector)
                  </span>
                  <span>
                    <i className="legend-dot issue" /> Bất đồng cần xác nhận
                  </span>
                </div>
              </div>

              {/* Closed-Loop Resolution Issues Cards */}
              <div className="qc-issues-section">
                <div className="qc-issues-heading">
                  <h3>
                    Bất đồng phát hiện trên frame này (
                    {selectedSample.qc_issues?.length || 0} nghi vấn)
                  </h3>
                  <p className="muted small">
                    Đối chiếu nhãn và bấm 1-Click để tự động sửa Ground Truth hoặc bác bỏ cảnh báo sai của model.
                  </p>
                </div>

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
                    {selectedSample.qc_issues.map((issue) => {
                      const isPending = issue.status === "PENDING";
                      const isResolved = issue.status === "RESOLVED";
                      const isAccepted = issue.status === "ACCEPTED";
                      const isRejected = issue.status === "REJECTED" || issue.status === "FALSE_POSITIVE";

                      return (
                        <div
                          key={issue.id}
                          className={`qc-issue-card ${
                            issue.issue_type === "MISSING_OBJECT" ? "missing" : "wrong-class"
                          } ${!isPending ? "resolved" : ""}`}
                        >
                          <div className="qc-issue-header">
                            <div className="qc-issue-type-badge">
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
                            </div>

                            <div className="qc-issue-badges">
                              <span className="badge badge-accent">
                                QC Score: {issue.qc_score.toFixed(3)}
                              </span>
                              <span
                                className={`badge ${
                                  isPending
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

                          {/* Comparison Visual Panel */}
                          <div className="qc-comparison-box">
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
    </div>
  );
}
