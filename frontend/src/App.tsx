import { useEffect, useState } from "react";
import {
  House,
  LayoutDashboard,
  Database,
  ChartNoAxesCombined,
  Shuffle,
  Users,
  ListChecks,
  Settings as SettingsIcon,
  ScanLine,
  Upload,
  ArrowRight,
  ChevronRight,
  Download,
  Check,
  AlertTriangle,
  X,
  LoaderCircle,
  Eye,
  Trash2,
  RotateCcw,
  Plus,
  Search,
  ShieldAlert,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { api, json } from "./api";
import { displayLabel } from "./uiText";
import type { Dataset, Sample, Run, Reviewer, Settings } from "./types";
import {
  usePageNavigation,
  isDatasetPage,
  type Page,
} from "./hooks/usePageNavigation";
import {
  DatasetRequiredDialog,
  DatasetRequiredRoute,
  type DatasetStatus,
} from "./components/DatasetRequiredRoute";
import { HomePage } from "./components/HomePage";
import { AppearanceSettings } from "./components/AppearanceSettings";
import { ModelQCPage } from "./components/ModelQCPage";

const nav = [
  ["Home", House],
  ["Model QC", ShieldAlert],
  ["Dataset", Database],
  ["Dashboard", LayoutDashboard],
  ["Settings", SettingsIcon],
] as const;
const colors = ["#38bdf8", "#818cf8", "#2dd4bf", "#fb923c", "#f472b6"];
const fmt = (n: number | null | undefined) =>
  n == null ? "Chưa có dữ liệu" : n.toFixed(1);
const title = displayLabel;
const Badge = ({ level }: { level: string }) => (
  <span className={"badge " + level.toLowerCase()}>{title(level)}</span>
);
const Empty = ({ text }: { text: string }) => (
  <div className="empty">
    <ScanLine size={30} />
    <p>{text}</p>
  </div>
);

function Chart({
  data,
  pie = false,
}: {
  data: { name: string; value: number }[];
  pie?: boolean;
}) {
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        {pie ? (
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={58}
              outerRadius={86}
              paddingAngle={3}
            >
              {data.map((_, i) => (
                <Cell key={i} fill={colors[i % colors.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "var(--surface-panel)",
                border: "1px solid var(--border)",
                color: "var(--text-main)",
              }}
            />
          </PieChart>
        ) : (
          <BarChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              vertical={false}
            />
            <XAxis
              dataKey="name"
              tick={{ fill: "var(--text-muted)", fontSize: "0.79rem" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "var(--text-muted)", fontSize: "0.79rem" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "var(--surface-panel)",
                border: "1px solid var(--border)",
                color: "var(--text-main)",
              }}
            />
            <Bar dataKey="value" fill="#38bdf8" radius={[5, 5, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function SampleTable({
  rows,
  onOpen,
  review = false,
}: {
  rows: Sample[];
  onOpen: (s: Sample) => void;
  review?: boolean;
}) {
  return rows.length ? (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {review && <th>Ưu tiên</th>}
            <th>Ảnh / mẫu</th>
            <th>Loại dữ liệu</th>
            <th>Nhãn</th>
            <th>Đặc điểm ảnh</th>
            <th>Đặc điểm nhãn</th>
            <th>Độ khó</th>
            <th>Mức độ</th>
            {rows.some((r) => r.sampling_reason) && <th>Cách chọn</th>}
            {review && <th>Trạng thái</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((s, i) => (
            <tr key={s.id} onClick={() => onOpen(s)}>
              {review && (
                <td className="muted">{String(i + 1).padStart(2, "0")}</td>
              )}
              <td>
                <div className="file-cell">
                  {s.media_url &&
                  [".jpg", ".jpeg", ".png"].includes(s.media_kind || "") ? (
                    <img src={s.media_url} alt="" />
                  ) : (
                    <div className="thumb">
                      <ScanLine size={18} />
                    </div>
                  )}
                  <div>
                    <strong>{s.file_name}</strong>
                    <small>
                      #{s.id}
                      {s.rare_classes.length > 0 ? " · Có loại vật thể hiếm" : ""}
                    </small>
                  </div>
                </div>
              </td>
              <td>{title(s.task_type)}</td>
              <td>{s.annotation_count}</td>
              <td>{fmt(s.visual_difficulty)}</td>
              <td>{fmt(s.annotation_difficulty)}</td>
              <td>
                <strong className="score">{fmt(s.overall_difficulty)}</strong>
              </td>
              <td>
                <Badge level={s.level} />
              </td>
              {rows.some((r) => r.sampling_reason) && (
                <td>{title(s.sampling_reason || "")}</td>
              )}
              {review && <td>{title(s.status || "PENDING")}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <Empty
      text={
        review
          ? "Người kiểm tra này chưa được phân công mẫu nào."
          : "Không có mẫu phù hợp với bộ lọc."
      }
    />
  );
}

export default function App() {
  const { page, navigate } = usePageNavigation();
  const [isNavOpen, setIsNavOpen] = useState<boolean>(true);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>("loading");
  const [guardFeature, setGuardFeature] = useState<Page | null>(null);
  const [pendingFeature, setPendingFeature] = useState<Page | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]),
    [active, setActive] = useState<number | null>(null),
    [samples, setSamples] = useState<Sample[]>([]),
    [run, setRun] = useState<Run | null>(null),
    [reviewers, setReviewers] = useState<Reviewer[]>([]),
    [assignments, setAssignments] = useState<Sample[]>([]),
    [settings, setSettings] = useState<Settings | null>(null);
  const [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [upload, setUpload] = useState(false),
    [detail, setDetail] = useState<Sample | null>(null),
    [reviewer, setReviewer] = useState<number | null>(null),
    [notice, setNotice] = useState("");
  const dataset = datasets.find((d) => d.id === active);
  async function load(id = active) {
    setDatasetStatus("loading");
    try {
      const [ds, config] = await Promise.all([
        api<Dataset[]>("/datasets"),
        api<Settings>("/settings").catch(() => null),
      ]);
      const selected =
        id && ds.some((d) => d.id === id) ? id : ds[0]?.id || null;
      if (selected) {
        const s = await api<Sample[]>(`/datasets/${selected}/samples`);
        setSamples(s);
        setRun(null);
        setReviewers([]);
        setAssignments([]);
        setReviewer(null);
      } else {
        setSamples([]);
        setRun(null);
        setReviewers([]);
        setAssignments([]);
      }
      setDatasets(ds);
      setActive(selected);
      setSettings(config);
      setDatasetStatus("ready");
      return selected;
    } catch (error) {
      setDatasetStatus("error");
      throw error;
    }
  }
  async function act(label: string, fn: () => Promise<unknown>) {
    setBusy(label);
    setError("");
    setNotice("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  useEffect(() => {
    void act("Đang tải không gian làm việc…", () => load());
  }, []);

  // Keyboard shortcut Ctrl+B / Cmd+B to toggle sidebar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) {
          return;
        }
        e.preventDefault();
        setIsNavOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    if (datasetStatus === "ready" && !dataset && isDatasetPage(page) && !upload)
      setGuardFeature(page);
    else if (dataset || datasetStatus !== "ready") setGuardFeature(null);
  }, [page, datasetStatus, !!dataset]);
  const requestPage = (next: Page) => {
    if (isDatasetPage(next) && datasetStatus === "ready" && !dataset)
      setGuardFeature(next);
    else navigate(next);
  };
  const cancelGuard = () => {
    setGuardFeature(null);
    if (isDatasetPage(page) && !dataset) navigate("Home", true);
  };
  const startUpload = (feature: Page | null = null) => {
    setPendingFeature(feature);
    setGuardFeature(null);
    setUpload(true);
  };
  const open = (s: Sample) => {
    void act("Đang tải ảnh và nhãn…", async () => {
      const data = await api<Sample>(`/samples/${s.id}`);
      setDetail({ ...data, ...(s.assignment_id ? s : {}) });
    });
  };
  const queue = assignments
    .filter((a) => a.reviewer_id === reviewer)
    .sort(
      (a, b) =>
        (b.overall_difficulty || 0) - (a.overall_difficulty || 0) ||
        a.id - b.id,
    );
  const next = () => {
    const s = queue.find((s) => s.status === "PENDING" && s.id !== detail?.id);
    if (s) open(s);
    else {
      setDetail(null);
      setNotice(
        "Đã kiểm tra hết danh sách. Các mẫu bỏ qua vẫn được giữ để kiểm tra sau.",
      );
    }
  };
  const exportLink = (kind: string) => (
    <a
      className="button secondary"
      href={`/api/datasets/${active}/exports/${kind}`}
    >
      <Download size={15} /> Tải bảng CSV
    </a>
  );
  return (
    <div className={`app ${isNavOpen ? "" : "nav-collapsed"}`}>
      <aside>
        {isNavOpen ? (
          <div className="aside-top-row">
            <div className="brand">
              <div className="brand-icon">
                <ScanLine size={20} />
              </div>
              <div className="brand-text">
                AnnoPilot<small>TRỢ LÝ KIỂM TRA NHÃN</small>
              </div>
            </div>
            <button
              className="nav-toggle-btn"
              onClick={() => setIsNavOpen(false)}
              title="Thu gọn thanh điều hướng (Ctrl+B)"
              aria-label="Thu gọn thanh điều hướng"
            >
              <PanelLeftClose size={18} />
            </button>
          </div>
        ) : (
          <>
            <div className="aside-top-row collapsed-brand-row">
              <button
                className="brand-icon-btn"
                onClick={() => setIsNavOpen(true)}
                title="AnnoPilot · Mở thanh điều hướng (Ctrl+B)"
                aria-label="Mở thanh điều hướng"
              >
                <div className="brand-icon">
                  <ScanLine size={20} />
                </div>
              </button>
            </div>
            <div className="nav-collapsed-divider" />
          </>
        )}

        {isNavOpen && <div className="nav-label">KHÔNG GIAN LÀM VIỆC</div>}
        <nav>
          {nav.map(([name, Icon]) => (
            <button
              key={name}
              onClick={() => requestPage(name)}
              className={page === name ? "active" : ""}
              aria-current={page === name ? "page" : undefined}
              title={title(name)}
            >
              <Icon size={19} />
              {isNavOpen && <span>{title(name)}</span>}
              {isNavOpen && page === name && <span className="nav-dot" />}
            </button>
          ))}
        </nav>

        {isNavOpen ? (
          <div className="sidebar-bottom">
            <span className="online-dot" /> Làm việc trên máy này
            <small>AI gợi ý, bạn quyết định.</small>
          </div>
        ) : (
          <div className="sidebar-collapsed-footer">
            <button
              className="nav-toggle-btn collapsed-expand-btn"
              onClick={() => setIsNavOpen(true)}
              title="Mở thanh điều hướng (Ctrl+B)"
              aria-label="Mở thanh điều hướng"
            >
              <PanelLeftOpen size={19} />
            </button>
            <div className="collapsed-status-dot" title="Máy chủ cục bộ · Đang kết nối" />
          </div>
        )}
      </aside>
      <div className="main">
        <header>
          <div className="breadcrumb">
            Không gian làm việc <ChevronRight size={14} />
            <strong>{title(page)}</strong>
          </div>
          <div className="header-actions">
            {dataset && (
              <select
                aria-label="Bộ dữ liệu đang chọn"
                value={active || ""}
                onChange={(e) =>
                  void act("Đang chuyển bộ dữ liệu…", () =>
                    load(Number(e.target.value)),
                  )
                }
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            )}
            <button onClick={() => startUpload()}>
              <Upload size={16} />
              Thêm bộ dữ liệu
            </button>
          </div>
        </header>
        <main>
          {error && (
            <div className="alert error" role="alert">
              <AlertTriangle size={18} />
              {error}
              <button className="icon-button" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="alert">
              <Check size={18} />
              {notice}
            </div>
          )}
          {busy && (
            <div className="busy" role="status">
              <LoaderCircle size={17} className="spin" />
              {busy}
            </div>
          )}
          {page === "Home" ? (
            <HomePage
              dataset={dataset}
              status={datasetStatus}
              onUpload={() => startUpload()}
              onDashboard={() => requestPage("Dataset")}
              onModelQC={() => requestPage("Model QC")}
            />
          ) : page !== "Settings" ? (
            <DatasetRequiredRoute
              status={datasetStatus}
              hasDataset={!!dataset}
              feature={page}
              onUpload={() => startUpload(page)}
              onCancel={cancelGuard}
              onRetry={() => void act("Đang kiểm tra bộ dữ liệu…", () => load())}
            >
              <>
                {dataset && page !== "Model QC" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <div className="eyebrow">
                          KIỂM SOÁT CHẤT LƯỢNG NHÃN
                        </div>
                        <h1>
                          {page === "Dashboard"
                            ? "Tổng quan bộ dữ liệu"
                            : page === "Workload"
                              ? "Phân công kiểm tra"
                              : title(page)}
                        </h1>
                        <p>
                          {page === "Smart Sampling"
                            ? "Chọn các mẫu cần kiểm tra trước."
                            : page === "Workload"
                              ? "Chia công việc cho từng người kiểm tra."
                              : page === "Review Queue"
                                ? "Xem các mẫu được ưu tiên kiểm tra."
                                : "Xem tình trạng dữ liệu và khối lượng cần kiểm tra."}
                        </p>
                      </div>
                      <div className="dataset-status">
                        <span className="online-dot" /> Đã phân tích
                        <small>
                          {dataset.name} · {dataset.format} ·{" "}
                          {title(dataset.task_type)}
                        </small>
                      </div>
                    </div>
                    <div className="principle">
                      <Eye size={16} />
                      Độ khó ước tính công sức kiểm tra ảnh. Muốn biết nhãn đúng hay sai, hãy mở mục Kiểm tra nhãn và xem từng nghi vấn.
                    </div>
                  </>
                )}
                {dataset && page === "Dashboard" && (
                  <>
                    <div className="stats">
                      <Metric
                        label="Tổng số mẫu"
                        value={dataset.sample_count}
                        note={`${dataset.missing_media} mẫu chưa có ảnh / dữ liệu gốc`}
                        icon={<Database />}
                      />
                      <Metric
                        label="Nhãn"
                        value={dataset.annotation_count}
                        note={`${Object.keys(dataset.classes).length} loại vật thể`}
                        icon={<ScanLine />}
                      />
                      <Metric
                        label="Độ khó trung bình"
                        value={fmt(dataset.average_difficulty)}
                        note="Công sức kiểm tra ước tính · thang 100"
                        icon={<ChartNoAxesCombined />}
                      />
                      <Metric
                        label="Mẫu khó"
                        value={dataset.levels.HARD || 0}
                        note={`${((100 * (dataset.levels.HARD || 0)) / Math.max(1, dataset.sample_count)).toFixed(1)}% bộ dữ liệu`}
                        icon={<AlertTriangle />}
                      />
                    </div>
                    <div className="grid two">
                      <section className="card">
                        <CardTitle
                          title="Phân bố độ khó"
                          sub="Số mẫu theo công sức kiểm tra ước tính"
                        />
                        <Chart
                          data={Array.from({ length: 10 }, (_, i) => ({
                            name: `${i * 10}–${i === 9 ? 100 : i * 10 + 9}`,
                            value: samples.filter(
                              (s) =>
                                s.overall_difficulty != null &&
                                s.overall_difficulty >= i * 10 &&
                                (i === 9
                                  ? s.overall_difficulty <= 100
                                  : s.overall_difficulty < (i + 1) * 10),
                            ).length,
                          }))}
                        />
                      </section>
                      <section className="card">
                        <CardTitle
                          title="Tỷ lệ mức độ khó"
                          sub="Dễ · Trung bình · Khó · Chưa có dữ liệu"
                        />
                        <Chart
                          pie
                          data={Object.entries(dataset.levels).map(
                            ([name, value]) => ({ name: title(name), value }),
                          )}
                        />
                        <div className="legend">
                          {Object.entries(dataset.levels).map(([k, v], i) => (
                            <span key={k}>
                              <i style={{ background: colors[i % 5] }} />
                              {title(k)} <strong>{v}</strong>
                            </span>
                          ))}
                        </div>
                      </section>
                      <section className="card">
                        <CardTitle
                          title="Phân bố loại vật thể"
                          sub="Số nhãn theo từng loại vật thể đã nhập"
                        />
                        {dataset.annotation_count ? (
                          <Chart
                            data={Object.entries(dataset.classes).map(
                              ([name, value]) => ({ name, value }),
                            )}
                          />
                        ) : (
                          <Empty text="Bộ dữ liệu chưa có nhãn hợp lệ." />
                        )}
                      </section>
                      <section className="card">
                        <CardTitle
                          title="Thông tin cần chú ý"
                          sub="Các đặc điểm đo được từ bộ dữ liệu"
                        />
                        <div className="insights">
                          <p>
                            <span>{dataset.rare_classes.length}</span> loại vật thể hiếm theo ngưỡng đã cài đặt.
                          </p>
                          <p>
                            <span>
                              {
                                samples.filter(
                                  (s) =>
                                    s.visual_difficulty != null &&
                                    s.visual_difficulty >= 70,
                                ).length
                              }
                            </span>{" "}
                            mẫu có đặc điểm ảnh khó kiểm tra.
                          </p>
                          <p>
                            <span>{dataset.missing_media}</span> mẫu chưa có tệp ảnh hoặc dữ liệu gốc.
                          </p>
                          <p>
                            <span>{dataset.levels.EASY || 0}</span> dễ ·{" "}
                            <span>{dataset.levels.MEDIUM || 0}</span> trung bình ·{" "}
                            <span>{dataset.levels.HARD || 0}</span> mẫu khó.
                          </p>
                          <p>
                            {Object.entries(dataset.annotation_types)
                              .map(([k, v]) => `${title(k)}: ${v}`)
                              .join(" · ") || "Chưa có hình dạng nhãn được hỗ trợ"}
                          </p>
                        </div>
                      </section>
                    </div>
                    <div className="grid two">
                      {dataset.annotation_count > 0 && (
                        <section className="card">
                          <CardTitle
                            title="Các dạng nhãn"
                            sub="Dạng nhãn được nhận diện trong bộ dữ liệu"
                          />
                          <Chart
                            data={Object.entries(dataset.annotation_types).map(
                              ([name, value]) => ({ name: title(name), value }),
                            )}
                          />
                        </section>
                      )}
                      {samples.some((s) => s.visual_difficulty !== null) && (
                        <section className="card">
                          <CardTitle
                            title="Độ khó theo đặc điểm ảnh"
                            sub="Chỉ tính ảnh có thể đọc được"
                          />
                          <Chart
                            data={Array.from({ length: 5 }, (_, i) => ({
                              name: `${i * 20}–${i === 4 ? 100 : i * 20 + 19}`,
                              value: samples.filter(
                                (s) =>
                                  s.visual_difficulty !== null &&
                                  s.visual_difficulty >= i * 20 &&
                                  (i === 4
                                    ? s.visual_difficulty <= 100
                                    : s.visual_difficulty < (i + 1) * 20),
                              ).length,
                            }))}
                          />
                        </section>
                      )}
                    </div>
                    {dataset.warnings.length > 0 && (
                      <details className="card warnings">
                        <summary>
                          <AlertTriangle size={16} />
                          {dataset.warnings.length} cảnh báo khi nhập và ghi chú dữ liệu
                        </summary>
                        <ul>
                          {dataset.warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <section className="card">
                      <CardTitle
                        title="Các mẫu cần nhiều công kiểm tra"
                        sub="Gợi ý bắt đầu kiểm tra; điểm khó chưa phải kết luận có lỗi"
                      />
                      <SampleTable
                        rows={[...samples]
                          .sort(
                            (a, b) =>
                              (b.overall_difficulty || 0) -
                              (a.overall_difficulty || 0),
                          )
                          .slice(0, 5)}
                        onOpen={open}
                      />
                    </section>
                  </>
                )}
                {dataset && (page === "Dataset" || page === "Difficulty") && (
                  <SampleBrowser
                    samples={samples}
                    onOpen={open}
                    difficulty={page === "Difficulty"}
                  />
                )}
                {dataset && page === "Smart Sampling" && (
                  <>
                    <SamplingForm
                      disabled={!!busy}
                      onGenerate={(config) =>
                        void act("Đang chọn mẫu kiểm tra…", async () => {
                          await api(
                            `/datasets/${active}/sampling`,
                            json("POST", config),
                          );
                          await load();
                        })
                      }
                    />
                    {run ? (
                      <section className="card">
                        <CardTitle
                          title={`Đã chọn ${run.selections.length} / ${dataset.sample_count} mẫu`}
                          sub={
                            run.needs_regeneration
                              ? "Cần tạo lại danh sách vì điểm đã thay đổi"
                              : "Chọn có thể lặp lại · Không trùng mẫu"
                          }
                          action={exportLink("smart_sample")}
                        />
                        {run.needs_regeneration && (
                          <div className="alert">Cần tạo lại</div>
                        )}
                        <div className="legend">
                          {[
                            "random",
                            "difficulty",
                            "edge",
                            "coverage_fallback",
                          ].map((k) => (
                            <span key={k}>
                              {title(k)}{" "}
                              <strong>
                                {
                                  run.selections.filter(
                                    (s) => s.sampling_reason === k,
                                  ).length
                                }
                              </strong>
                            </span>
                          ))}
                        </div>
                        <SampleTable rows={run.selections} onOpen={open} />
                      </section>
                    ) : (
                      <Empty text="Chọn số lượng hoặc tỷ lệ mẫu, rồi bấm Tạo danh sách kiểm tra." />
                    )}
                  </>
                )}
                {dataset && page === "Workload" && (
                  <>
                    <section className="card">
                      <CardTitle
                        title="Nhóm kiểm tra"
                        sub="Phân chia các mẫu đã chọn theo công sức kiểm tra ước tính."
                      />
                      <ReviewerForm
                        disabled={!!busy}
                        onAdd={(name) =>
                          void act("Đang thêm người kiểm tra…", async () => {
                            await api(
                              `/datasets/${active}/reviewers`,
                              json("POST", { name }),
                            );
                            await load();
                          })
                        }
                      />
                      <div className="reviewer-chips">
                        {reviewers.map((r) => (
                          <span key={r.id}>
                            {r.name}
                            <button
                              aria-label={`Remove ${r.name}`}
                              className="icon-button"
                              disabled={!!busy}
                              onClick={() =>
                                void act("Đang xóa người kiểm tra…", async () => {
                                  await api(`/reviewers/${r.id}`, {
                                    method: "DELETE",
                                  });
                                  await load();
                                })
                              }
                            >
                              <X size={14} />
                            </button>
                          </span>
                        ))}
                      </div>
                      <button
                        disabled={
                          !!busy ||
                          !reviewers.length ||
                          !run ||
                          run.needs_regeneration
                        }
                        onClick={() =>
                          void act("Đang phân công…", async () => {
                            await api(`/datasets/${active}/balance`, {
                              method: "POST",
                            });
                            await load();
                          })
                        }
                      >
                        <Users size={16} />
                        Tự động phân công
                      </button>
                      {!run && (
                        <p className="muted">
                          Hãy tạo danh sách ở mục Chọn mẫu kiểm tra trước.
                        </p>
                      )}
                      {run?.needs_regeneration && (
                        <p className="warning-text">
                          Cần tạo lại danh sách mẫu trước khi phân công.
                        </p>
                      )}
                    </section>
                    {reviewers.length > 0 && (
                      <div className="grid reviewer-grid">
                        {reviewers.map((r) => {
                          const rows = assignments.filter(
                              (a) => a.reviewer_id === r.id,
                            ),
                            total = rows.reduce(
                              (sum, s) => sum + (s.overall_difficulty || 0),
                              0,
                            );
                          const unavailable = rows.some(
                            (s) => s.overall_difficulty === null,
                          );
                          return (
                            <section className="card" key={r.id}>
                              <h3>{r.name}</h3>
                              <div className="big-number">
                                {unavailable ? "Chưa có dữ liệu" : total.toFixed(1)}
                                <small>Tổng điểm độ khó</small>
                              </div>
                              <p className="muted">
                                {rows.length} mẫu · Trung bình{" "}
                                {unavailable
                                  ? "Chưa có dữ liệu"
                                  : rows.length
                                    ? (total / rows.length).toFixed(1)
                                    : "—"}{" "}
                                ·{" "}
                                {rows.filter((s) => s.level === "HARD").length}{" "}
                                khó
                              </p>
                            </section>
                          );
                        })}
                      </div>
                    )}
                    {assignments.length > 0 && (
                      <>
                        <section className="card">
                          <CardTitle
                            title="Khối lượng kiểm tra ước tính"
                            sub="Tổng điểm độ khó của các mẫu được giao"
                          />
                          {assignments.some(
                            (s) => s.overall_difficulty === null,
                          ) ? (
                            <Empty text="Chưa có phân công phù hợp. Hãy tạo lại sau khi điều chỉnh trọng số." />
                          ) : (
                            <Chart
                              data={reviewers.map((r) => ({
                                name: r.name,
                                value: assignments
                                  .filter((a) => a.reviewer_id === r.id)
                                  .reduce(
                                    (sum, s) =>
                                      sum + (s.overall_difficulty || 0),
                                    0,
                                  ),
                              }))}
                            />
                          )}
                        </section>
                        <section className="card">
                          <CardTitle
                            title="Assignments"
                            sub={
                              assignments.some((a) => a.needs_regeneration)
                                ? "Cần tạo lại"
                                : "Sau khi lưu kết quả kiểm tra, hãy tạo danh sách mẫu mới để phân công lại."
                            }
                            action={exportLink("review_assignments")}
                          />
                          <div className="table-wrap">
                            <table>
                              <thead>
                                <tr>
                                  <th>Ảnh / mẫu</th>
                                  <th>Loại dữ liệu</th>
                                  <th>Độ khó</th>
                                  <th>Mức độ</th>
                                  <th>Người kiểm tra</th>
                                </tr>
                              </thead>
                              <tbody>
                                {assignments.map((s) => (
                                  <tr key={s.id} onClick={() => open(s)}>
                                    <td>{s.file_name}</td>
                                    <td>{title(s.task_type)}</td>
                                    <td>{fmt(s.overall_difficulty)}</td>
                                    <td>
                                      <Badge level={s.level} />
                                    </td>
                                    <td>{s.reviewer}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </section>
                      </>
                    )}
                  </>
                )}
                {dataset && page === "Review Queue" && (
                  <>
                    <section className="card">
                      <CardTitle
                        title="Danh sách kiểm tra theo người"
                        sub="Mẫu khó được xếp trước trong danh sách của mỗi người."
                        action={exportLink("review_results")}
                      />
                      <label>
                        Người kiểm tra
                        <select
                          value={reviewer || ""}
                          onChange={(e) => setReviewer(Number(e.target.value))}
                        >
                          <option value="" disabled>
                            Chọn người kiểm tra
                          </option>
                          {reviewers.map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      {queue.some((a) => a.needs_regeneration) && (
                        <div className="alert">
                          Cần tạo lại phân công vì điểm độ khó đã thay đổi.
                        </div>
                      )}
                      <div className="queue-stats">
                        {[
                          ["Tổng", queue.length],
                          [
                            "Đã kiểm tra",
                            queue.filter(
                              (s) =>
                                s.status === "REVIEWED" ||
                                s.status === "ISSUE_FOUND",
                            ).length,
                          ],
                          [
                            "Có lỗi",
                            queue.filter((s) => s.status === "ISSUE_FOUND")
                              .length,
                          ],
                          [
                            "Chưa kiểm tra",
                            queue.filter((s) => s.status === "PENDING").length,
                          ],
                          [
                            "Đã bỏ qua",
                            queue.filter((s) => s.status === "SKIPPED").length,
                          ],
                        ].map(([k, v]) => (
                          <div key={k}>
                            <strong>{v}</strong>
                            <span>{k}</span>
                          </div>
                        ))}
                      </div>
                      <SampleTable rows={queue} onOpen={open} review />
                    </section>
                  </>
                )}
                {dataset && page === "Model QC" && (
                  <ModelQCPage
                    dataset={dataset}
                    busy={!!busy}
                    act={act}
                    onRefreshDataset={async () => {
                      await load();
                    }}
                  />
                )}
                {dataset && page === "Dataset" && (
                  <div className="footer-actions">
                    <button
                      className="secondary"
                      disabled={!!busy}
                      onClick={() =>
                        void act("Đang tính lại độ khó…", async () => {
                          await api(`/datasets/${active}/recalculate`, {
                            method: "POST",
                          });
                          await load();
                        })
                      }
                    >
                      <RotateCcw size={15} />
                      Tính lại độ khó
                    </button>
                    <button
                      className="danger secondary"
                      disabled={!!busy}
                      onClick={() => {
                        if (
                          window.confirm(
                            `Xóa bộ dữ liệu “${dataset.name}” cùng kết quả phân tích, danh sách mẫu và kết quả kiểm tra? Thao tác này không thể hoàn tác.`,
                          )
                        )
                          void act("Đang xóa bộ dữ liệu…", async () => {
                            await api(`/datasets/${active}`, {
                              method: "DELETE",
                            });
                            navigate("Home");
                            await load(null);
                          });
                      }}
                    >
                      <Trash2 size={15} />
                      Xóa bộ dữ liệu
                    </button>
                  </div>
                )}
              </>
            </DatasetRequiredRoute>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">TÙY CHỈNH KHÔNG GIAN LÀM VIỆC</div>
                  <h1>Cài đặt</h1>
                  <p>Chọn giao diện, cỡ chữ và cách tính độ khó.</p>
                </div>
              </div>
              <AppearanceSettings />
              {settings ? (
                <SettingsForm
                  settings={settings}
                  disabled={!!busy}
                  onSave={(s) =>
                    void act("Đang tính lại độ khó…", async () => {
                      await api("/settings", json("PUT", s));
                      await load();
                      setNotice(
                        "Đã lưu cài đặt và tính lại độ khó. Hãy tạo lại danh sách mẫu và phân công hiện có.",
                      );
                    })
                  }
                />
              ) : datasetStatus === "error" ? (
                <section className="card" role="alert">
                  <h2>Chưa tải được cài đặt tính điểm</h2>
                  <p>
                    Bạn vẫn có thể đổi giao diện. Kết nối lại máy chủ để tải cài đặt tính điểm.
                  </p>
                  <button
                    onClick={() =>
                      void act("Đang kiểm tra bộ dữ liệu…", () => load())
                    }
                  >
                    Thử lại
                  </button>
                </section>
              ) : (
                <div className="busy" role="status">
                  <LoaderCircle className="spin" size={18} />
                  Đang tải cài đặt tính điểm…
                </div>
              )}
            </>
          )}
        </main>
        <footer>
          AnnoPilot <span>Trợ lý kiểm tra chất lượng nhãn</span>
          <span>Xử lý trên máy chủ · AI hỗ trợ · Bạn quyết định</span>
        </footer>
      </div>
      {guardFeature && (
        <DatasetRequiredDialog
          feature={guardFeature}
          onCancel={cancelGuard}
          onUpload={() => startUpload(guardFeature)}
        />
      )}
      {upload && (
        <UploadDialog
          error={error}
          disabled={!!busy}
          onClose={() => setUpload(false)}
          onSubmit={(form) =>
            void act("Đang tải lên và phân tích bộ dữ liệu…", async () => {
              const d = await api<Dataset>("/datasets/upload", {
                method: "POST",
                body: form,
              });
              await load(d.id);
              navigate(pendingFeature || "Dashboard");
              setPendingFeature(null);
              setUpload(false);
            })
          }
        />
      )}
      {detail && (
        <Detail
          error={error}
          sample={detail}
          disabled={!!busy}
          onClose={() => setDetail(null)}
          onNext={next}
          onReview={(status, note) =>
            void act("Đang lưu kết quả kiểm tra…", async () => {
              const saved = await api<Sample>(
                `/assignments/${detail.assignment_id}/review`,
                json("PUT", { status, note }),
              );
              setDetail({ ...detail, ...saved });
              await load();
            })
          }
        />
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  note,
  icon,
}: {
  label: string;
  value: React.ReactNode;
  note: string;
  icon: React.ReactNode;
}) {
  return (
    <section className="card metric">
      <div>
        {label}
        <span>{icon}</span>
      </div>
      <strong>{value}</strong>
      <small>{note}</small>
    </section>
  );
}
function CardTitle({
  title,
  sub,
  action,
}: {
  title: string;
  sub: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="card-title">
      <div>
        <h3>{title}</h3>
        <p>{sub}</p>
      </div>
      {action}
    </div>
  );
}
function SampleBrowser({
  samples,
  onOpen,
  difficulty,
}: {
  samples: Sample[];
  onOpen: (s: Sample) => void;
  difficulty: boolean;
}) {
  const [search, setSearch] = useState(""),
    [level, setLevel] = useState(""),
    [task, setTask] = useState(""),
    [rare, setRare] = useState(false),
    [sort, setSort] = useState("desc"),
    [page, setPage] = useState(1);
  const rows = samples
    .filter(
      (s) =>
        s.file_name.toLowerCase().includes(search.toLowerCase()) &&
        (!level || s.level === level) &&
        (!task || s.task_type === task) &&
        (!rare || s.rare_classes.length > 0),
    )
    .sort((a, b) =>
      sort === "name"
        ? a.file_name.localeCompare(b.file_name)
        : sort === "asc"
          ? (a.overall_difficulty || 0) - (b.overall_difficulty || 0)
          : (b.overall_difficulty || 0) - (a.overall_difficulty || 0),
    );
  useEffect(() => setPage(1), [search, level, task, rare, sort, samples]);
  return (
    <section className="card">
      <CardTitle
        title={difficulty ? "Xem mẫu theo độ khó" : "Các mẫu đã nhập"}
        sub={`${rows.length} mẫu · Bấm một dòng để xem ảnh, nhãn và cách tính độ khó.`}
      />
      <div className="filters">
        <div className="search">
          <Search size={16} />
          <input
            aria-label="Tìm mẫu"
            placeholder="Nhập tên tệp để tìm…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <select
          aria-label="Lọc theo độ khó"
          value={level}
          onChange={(e) => setLevel(e.target.value)}
        >
          <option value="">Mọi mức độ</option>
          {["EASY", "MEDIUM", "HARD", "UNAVAILABLE"].map((v) => (
            <option key={v} value={v}>{title(v)}</option>
          ))}
        </select>
        <select
          aria-label="Lọc theo loại dữ liệu"
          value={task}
          onChange={(e) => setTask(e.target.value)}
        >
          <option value="">Mọi loại dữ liệu</option>
          {Array.from(new Set(samples.map((s) => s.task_type))).map((v) => (
            <option key={v} value={v}>{title(v)}</option>
          ))}
        </select>
        <select
          aria-label="Sắp xếp mẫu"
          value={sort}
          onChange={(e) => setSort(e.target.value)}
        >
          <option value="desc">Khó nhất trước</option>
          <option value="asc">Dễ nhất trước</option>
          <option value="name">Tên tệp</option>
        </select>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={rare}
            onChange={(e) => setRare(e.target.checked)}
          />
          Chỉ mẫu có loại vật thể hiếm
        </label>
      </div>
      <SampleTable
        rows={rows.slice((page - 1) * 20, page * 20)}
        onOpen={onOpen}
      />
      <div className="pagination">
        <span>
          {rows.length
            ? `${(page - 1) * 20 + 1}–${Math.min(page * 20, rows.length)} / ${rows.length} mẫu`
            : "0 mẫu"}
        </span>
        <button
          className="secondary"
          disabled={page === 1}
          onClick={() => setPage(page - 1)}
        >
          Trang trước
        </button>
        <button
          className="secondary"
          disabled={page * 20 >= rows.length}
          onClick={() => setPage(page + 1)}
        >
          Trang sau
        </button>
      </div>
    </section>
  );
}
function SamplingForm({
  disabled,
  onGenerate,
}: {
  disabled: boolean;
  onGenerate: (c: unknown) => void;
}) {
  const [mode, setMode] = useState("percentage"),
    [budget, setBudget] = useState(10),
    [seed, setSeed] = useState(42),
    [weights, setWeights] = useState({ random: 40, difficulty: 40, edge: 20 });
  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  return (
    <form
      className="card"
      onSubmit={(e) => {
        e.preventDefault();
        onGenerate({
          mode,
          budget,
          seed,
          weights: Object.fromEntries(
            Object.entries(weights).map(([k, v]) => [k, v / 100]),
          ),
        });
      }}
    >
      <CardTitle
        title="Chọn mẫu để kiểm tra"
        sub="Kết hợp mẫu ngẫu nhiên, mẫu khó và trường hợp hiếm thành một danh sách kiểm tra."
      />
      <div className="form-grid">
        <label>
          Cách giới hạn số mẫu
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="percentage">Theo phần trăm</option>
            <option value="count">Theo số lượng mẫu</option>
          </select>
        </label>
        <label>
          Số lượng / tỷ lệ cần kiểm tra
          <input
            type="number"
            min="1"
            max={mode === "percentage" ? 100 : undefined}
            required
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
          />
        </label>
        <label>
          Mã lặp lại cách chọn (seed)
          <input
            type="number"
            required
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="form-grid">
        {Object.entries(weights).map(([k, v]) => (
          <label key={k}>
            {k === "random"
              ? "Mẫu ngẫu nhiên"
              : k === "difficulty"
                ? "Mẫu khó"
                : "Trường hợp hiếm"}{" "}
            (%)
            <input
              required
              type="number"
              min="0"
              max="100"
              value={v}
              onChange={(e) =>
                setWeights({ ...weights, [k]: Number(e.target.value) })
              }
            />
          </label>
        ))}
      </div>
      <div className="form-bottom">
        <span className={total === 100 ? "muted" : "warning-text"}>
          Tổng tỷ lệ: {total}% / 100%
        </span>
        <button disabled={disabled || total !== 100}>
          <Shuffle size={16} />
          Tạo danh sách kiểm tra
        </button>
      </div>
    </form>
  );
}
function ReviewerForm({
  disabled,
  onAdd,
}: {
  disabled: boolean;
  onAdd: (s: string) => void;
}) {
  const [name, setName] = useState("");
  return (
    <form
      className="inline-form"
      onSubmit={(e) => {
        e.preventDefault();
        onAdd(name);
        setName("");
      }}
    >
      <input
        required
        maxLength={100}
        placeholder="Tên người kiểm tra"
        aria-label="Tên người kiểm tra"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <button disabled={disabled || !name.trim()}>
        <Plus size={16} />
        Thêm người kiểm tra
      </button>
    </form>
  );
}
function SettingsForm({
  settings,
  disabled,
  onSave,
}: {
  settings: Settings;
  disabled: boolean;
  onSave: (s: Settings) => void;
}) {
  const [draft, setDraft] = useState(settings);
  useEffect(() => setDraft(settings), [settings]);
  const valid = Object.values(draft).every(
    (v) =>
      typeof v === "number" ||
      Math.abs(Object.values(v).reduce((a, b) => a + b, 0) - 1) < 0.000001,
  );
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSave(draft);
      }}
    >
      <div className="page-heading">
        <div>
          <div className="eyebrow">CÁCH TÍNH ĐỘ KHÓ</div>
          <h2>Cài đặt tính độ khó</h2>
          <p>
            Khi lưu, ứng dụng tính lại độ khó cho mọi bộ dữ liệu. Danh sách mẫu và phân công cũ cần tạo lại.
          </p>
        </div>
        <button disabled={disabled || !valid}>
          <RotateCcw size={16} />
          Lưu và tính lại
        </button>
      </div>
      <div className="grid two">
        {Object.entries(draft).map(([group, value]) => (
          <section className="card" key={group}>
            <CardTitle
              title={title(group)}
              sub={
                typeof value === "number"
                  ? "Ngưỡng tỷ lệ từ 0 đến 1"
                  : "Tổng trọng số phải bằng 1. Đặc điểm thiếu dữ liệu được bỏ qua và cân lại trọng số."
              }
            />
            {typeof value === "number" ? (
              <input
                type="number"
                min="0.000001"
                max="1"
                step="any"
                required
                value={value}
                onChange={(e) =>
                  setDraft({ ...draft, [group]: Number(e.target.value) })
                }
              />
            ) : (
              <div className="weight-fields">
                {Object.entries(value).map(([k, v]) => (
                  <label key={k}>
                    {title(k)}
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="any"
                      required
                      value={v}
                      onChange={(e) =>
                        setDraft({
                          ...draft,
                          [group]: { ...value, [k]: Number(e.target.value) },
                        })
                      }
                    />
                  </label>
                ))}
                <p
                  className={
                    Math.abs(
                      Object.values(value).reduce((a, b) => a + b, 0) - 1,
                    ) < 0.000001
                      ? "muted"
                      : "warning-text"
                  }
                >
                  Tổng{" "}
                  {Object.values(value)
                    .reduce((a, b) => a + b, 0)
                    .toFixed(4)}
                </p>
              </div>
            )}
          </section>
        ))}
      </div>
    </form>
  );
}
function UploadDialog({
  error,
  disabled,
  onClose,
  onSubmit,
}: {
  error: string;
  disabled: boolean;
  onClose: () => void;
  onSubmit: (f: FormData) => void;
}) {
  return (
    <div className="modal-backdrop">
      <section
        className="modal upload-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
      >
        <div className="modal-heading">
          <div>
            <div className="eyebrow">BẮT ĐẦU VỚI DỮ LIỆU</div>
            <h2 id="upload-title">Thêm bộ dữ liệu</h2>
          </div>
          <button
            aria-label="Đóng cửa sổ thêm dữ liệu"
            className="icon-button"
            disabled={disabled}
            onClick={onClose}
          >
            <X />
          </button>
        </div>
        <p className="muted">
          Thêm tệp nhãn và ảnh tương ứng. Chưa có nhãn? Tạo tác vụ chứa ảnh trên CVAT, xuất gói CVAT for images kèm ảnh (nhãn có thể rỗng), rồi nhập vào đây để tạo nhãn gợi ý bằng AI.
        </p>
        {error && (
          <div className="alert error" role="alert">
            {error}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit(new FormData(e.currentTarget));
          }}
        >
          <label>
            Tên bộ dữ liệu
            <input
              name="name"
              placeholder="Ví dụ: Ảnh giao thông cần kiểm tra"
              required
              maxLength={200}
            />
          </label>
          <div className="form-grid">
            <label>
              Dạng nhãn
              <select name="task_type">
                <option value="AUTO">Tự nhận diện</option>
                <option value="BBOX_2D">Khung bao vật thể (2D)</option>
                <option value="SEGMENTATION">Vùng phân đoạn</option>
                <option value="KEYPOINT">Điểm đặc trưng / tư thế</option>
                <option value="CUBOID_3D">Khối hộp (3D)</option>
              </select>
            </label>
            <label>
              Định dạng tệp nhãn
              <select name="format">
                <option value="AUTO">Tự nhận diện</option>
                {["CVAT", "COCO", "YOLO", "KITTI"].map((f) => (
                  <option key={f}>{f}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="file-drop">
            <Upload size={24} />
            <strong>Tệp nhãn hoặc ZIP chứa nhãn</strong>
            <span>Hỗ trợ CVAT XML · COCO JSON · YOLO / KITTI TXT</span>
            <input
              type="file"
              name="annotations"
              multiple
              required
              accept=".xml,.json,.txt,.zip"
            />
          </label>
          <label className="file-drop">
            <Database size={24} />
            <strong>Ảnh / dữ liệu cảm biến / tệp ZIP</strong>
            <span>JPG · JPEG · PNG · BIN · PCD (giữ lại tệp cảm biến)</span>
            <input
              type="file"
              name="media"
              multiple
              accept=".jpg,.jpeg,.png,.bin,.pcd,.zip"
            />
          </label>
          <p className="muted small">
            Nhãn YOLO cần có ảnh tương ứng. Hiện chưa tính mật độ điểm và phép chiếu KITTI do cần dữ liệu hiệu chỉnh cảm biến.
          </p>
          <button className="full" disabled={disabled}>
            {disabled ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Upload size={16} />
            )}
            Thêm dữ liệu và phân tích
          </button>
          {disabled && (
            <p className="muted">
              Đang tải tệp, đọc nhãn, ghép ảnh và tính độ khó. Vui lòng giữ cửa sổ này mở.
            </p>
          )}
        </form>
      </section>
    </div>
  );
}
function Detail({
  error,
  sample: s,
  disabled,
  onClose,
  onReview,
  onNext,
}: {
  error: string;
  sample: Sample;
  disabled: boolean;
  onClose: () => void;
  onReview: (status: string, note: string) => void;
  onNext: () => void;
}) {
  const [note, setNote] = useState(s.note || "");
  useEffect(() => setNote(s.note || ""), [s.id, s.note]);
  const image =
    s.media_url && [".jpg", ".jpeg", ".png"].includes(s.media_kind || "");
  return (
    <div className="modal-backdrop">
      <section
        className="modal detail-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="detail-title"
      >
        <div className="modal-heading">
          <div>
            <div className="eyebrow">
              MẪU #{s.id} · {title(s.task_type)}
            </div>
            <h2 id="detail-title">{s.file_name}</h2>
          </div>
          <button
            aria-label="Đóng chi tiết mẫu"
            className="icon-button"
            onClick={onClose}
          >
            <X />
          </button>
        </div>
        {error && (
          <div className="alert error" role="alert">
            {error}
          </div>
        )}
        <div className="detail-grid">
          <div>
            <div className="preview">
              {image ? (
                <div
                  className="image-stage"
                  style={{ aspectRatio: `${s.width || 1}/${s.height || 1}` }}
                >
                  <img src={s.media_url!} alt={s.file_name} />
                  <svg
                    viewBox={`0 0 ${s.width || 1} ${s.height || 1}`}
                    preserveAspectRatio="none"
                  >
                    {s.annotations?.map((a, i) => {
                      const color = colors[i % colors.length],
                        g = a.geometry;
                      const text = (x: number, y: number) => (
                        <text
                          x={x}
                          y={Math.max(14, y - 5)}
                          fill={color}
                          fontSize={Math.max(12, (s.width || 600) / 65)}
                          stroke="#07111e"
                          strokeWidth=".5"
                        >
                          {a.label}
                        </text>
                      );
                      if (a.shape_type === "BBOX_2D")
                        return (
                          <g key={a.id}>
                            <rect
                              x={g.x1}
                              y={g.y1}
                              width={g.x2 - g.x1}
                              height={g.y2 - g.y1}
                              fill={color + "12"}
                              stroke={color}
                              strokeWidth="2"
                              vectorEffect="non-scaling-stroke"
                            />
                            {text(g.x1, g.y1)}
                          </g>
                        );
                      if (a.shape_type === "POLYGON")
                        return (
                          <g key={a.id}>
                            <polygon
                              points={g.points
                                .map((p: number[]) => p.join(","))
                                .join(" ")}
                              fill={color + "25"}
                              stroke={color}
                              strokeWidth="2"
                              vectorEffect="non-scaling-stroke"
                            />
                            {text(g.points[0][0], g.points[0][1])}
                          </g>
                        );
                      if (a.shape_type === "KEYPOINT")
                        return (
                          <g key={a.id}>
                            {g.points
                              .filter((p: any) => p.visibility !== 0)
                              .map((p: any, j: number) => (
                                <circle
                                  key={j}
                                  cx={p.x}
                                  cy={p.y}
                                  r={Math.max(3, (s.width || 600) / 150)}
                                  fill={color}
                                  opacity={p.visibility === 1 ? 0.5 : 1}
                                />
                              ))}
                            {g.points.some((p: any) => p.visibility !== 0) &&
                              text(
                                g.points.find((p: any) => p.visibility !== 0).x,
                                g.points.find((p: any) => p.visibility !== 0).y,
                              )}
                          </g>
                        );
                      return null;
                    })}
                  </svg>
                </div>
              ) : (
                <Empty
                  text={
                    s.media_url
                      ? "Đã giữ tệp cảm biến. Xem thông tin khối hộp bên dưới; hiện chưa hỗ trợ xem đám mây điểm tương tác."
                      : "Chưa có ảnh / dữ liệu gốc"
                  }
                />
              )}
            </div>
            <div className="principle">
              <Eye size={15} />
              Độ khó ước tính công sức kiểm tra. Bạn cần xem nhãn để xác nhận lỗi.
            </div>
            <section className="card">
              <CardTitle
                title="Thông tin nhãn"
                sub={`${s.annotation_count} normalized shapes`}
              />
              {s.annotations?.map((a) => (
                <details key={a.id}>
                  <summary>
                    <Badge level={a.shape_type} />
                    {a.label}
                  </summary>
                  <pre>
                    {JSON.stringify(
                      {
                        geometry: a.geometry,
                        attributes: a.attributes,
                        occluded: a.occluded,
                        metadata: a.metadata,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              ))}
            </section>
          </div>
          <div>
            <section className="card">
              <CardTitle
                title="Cách tính điểm độ khó"
                sub="Điểm theo quy tắc cố định · thang 0–100"
              />
              <div className="detail-score">
                <strong>{fmt(s.overall_difficulty)}</strong>
                <Badge level={s.level} />
              </div>
              <div className="score-pair">
                <span>
                  Đặc điểm nhãn <strong>{fmt(s.annotation_difficulty)}</strong>
                </span>
                <span>
                  Đặc điểm ảnh <strong>{fmt(s.visual_difficulty)}</strong>
                </span>
              </div>
              <h4>Mức đóng góp của từng đặc điểm vào điểm tổng</h4>
              {Object.entries(s.analysis?.feature_contributions || {})
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => (
                  <div className="contribution" key={k}>
                    <span>{title(k)}</span>
                    <strong>+{v.toFixed(2)}</strong>
                    <div style={{ width: `${Math.min(v, 100)}%` }} />
                  </div>
                ))}
              <h4>Đặc điểm có dữ liệu và giá trị đo</h4>
              {[
                ...Object.entries(s.analysis?.visual.features || {}).map(
                  ([k, v]) => ["Đặc điểm ảnh · " + title(k), v] as const,
                ),
                ...Object.entries(s.analysis?.per_type || {}).flatMap(
                  ([kind, v]) =>
                    Object.entries(v.features).map(
                      ([k, f]) => [title(kind) + " · " + title(k), f] as const,
                    ),
                ),
              ].map(([k, f]) => (
                <details className="feature" key={k}>
                  <summary>
                    {title(k)}
                    <strong>
                      {f.available
                        ? `${((f.value || 0) * 100).toFixed(1)}%`
                        : "Chưa có dữ liệu"}
                    </strong>
                  </summary>
                  <p>
                    {f.reason_if_unavailable ||
                      "Giá trị độ khó đã chuẩn hóa"}
                  </p>
                  {f.raw != null && <pre>{JSON.stringify(f.raw, null, 2)}</pre>}
                </details>
              ))}
              {s.rare_classes.length > 0 && (
                <p>Loại vật thể hiếm: {s.rare_classes.join(", ")}</p>
              )}
            </section>
            {s.assignment_id && (
              <section className="card">
                <CardTitle
                  title="Ghi nhận kiểm tra"
                  sub={`${s.reviewer} · ${title(s.status || "PENDING")} · ${title(s.sampling_reason || "")}`}
                />
                <label>
                  Ghi chú kiểm tra
                  <textarea
                    rows={4}
                    maxLength={10000}
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Mô tả điều bạn quan sát được…"
                  />
                </label>
                <div className="review-buttons">
                  <button
                    disabled={disabled}
                    onClick={() => onReview("REVIEWED", note)}
                  >
                    <Check size={16} />
                    Chưa thấy lỗi
                  </button>
                  <button
                    className="issue"
                    disabled={disabled}
                    onClick={() => onReview("ISSUE_FOUND", note)}
                  >
                    <AlertTriangle size={16} />
                    Phát hiện lỗi
                  </button>
                  <button
                    className="secondary"
                    disabled={disabled}
                    onClick={() => onReview("SKIPPED", note)}
                  >
                    Bỏ qua lúc này
                  </button>
                  <button
                    className="secondary"
                    disabled={disabled}
                    onClick={() => onReview(s.status || "PENDING", note)}
                  >
                    Lưu ghi chú
                  </button>
                </div>
                <button
                  className="full secondary"
                  disabled={disabled}
                  onClick={onNext}
                >
                  Mở mẫu ưu tiên tiếp theo
                  <ArrowRight size={16} />
                </button>
              </section>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
