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
  ["Dashboard", LayoutDashboard],
  ["Model QC", ShieldAlert],
  ["Dataset", Database],
  ["Difficulty", ChartNoAxesCombined],
  ["Smart Sampling", Shuffle],
  ["Workload", Users],
  ["Review Queue", ListChecks],
  ["Settings", SettingsIcon],
] as const;
const colors = ["#38bdf8", "#818cf8", "#2dd4bf", "#fb923c", "#f472b6"];
const fmt = (n: number | null | undefined) =>
  n == null ? "Unavailable" : n.toFixed(1);
const title = (s: string) => s.replaceAll("_", " ").replaceAll(".", " · ");
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
            {review && <th>Priority</th>}
            <th>Sample</th>
            <th>Task</th>
            <th>Annotations</th>
            <th>Visual</th>
            <th>Annotation</th>
            <th>Difficulty</th>
            <th>Level</th>
            {rows.some((r) => r.sampling_reason) && <th>Selection</th>}
            {review && <th>Status</th>}
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
                      {s.rare_classes.length > 0 ? " · Rare class" : ""}
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
          ? "No assignments in this reviewer’s queue."
          : "No samples match this view."
      }
    />
  );
}

export default function App() {
  const { page, navigate } = usePageNavigation();
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
      const ds = await api<Dataset[]>("/datasets");
      const selected =
        id && ds.some((d) => d.id === id) ? id : ds[0]?.id || null;
      const config = await api<Settings>("/settings");
      if (selected) {
        const [s, r, rv, a] = await Promise.all([
          api<Sample[]>(`/datasets/${selected}/samples`),
          api<Run | null>(`/datasets/${selected}/sampling`),
          api<Reviewer[]>(`/datasets/${selected}/reviewers`),
          api<Sample[]>(`/datasets/${selected}/assignments`),
        ]);
        setSamples(s);
        setRun(r);
        setReviewers(rv);
        setAssignments(a);
        setReviewer((old) =>
          rv.some((r) => r.id === old) ? old : rv[0]?.id || null,
        );
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
    void act("Loading workspace…", () => load());
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
    void act("Loading sample…", async () => {
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
        "Review queue completed. Skipped samples remain available in the queue.",
      );
    }
  };
  const exportLink = (kind: string) => (
    <a
      className="button secondary"
      href={`/api/datasets/${active}/exports/${kind}`}
    >
      <Download size={15} /> Export CSV
    </a>
  );
  return (
    <div className="app">
      <aside>
        <button
          className="brand"
          aria-label="AnnoPilot Home"
          onClick={() => navigate("Home")}
        >
          <div className="brand-icon">
            <ScanLine />
          </div>
          <div>
            AnnoPilot<small>REVIEW INTELLIGENCE</small>
          </div>
        </button>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {nav.map(([name, Icon]) => (
            <button
              key={name}
              onClick={() => requestPage(name)}
              className={page === name ? "active" : ""}
              aria-current={page === name ? "page" : undefined}
            >
              <Icon size={19} />
              {name}
              {page === name && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="online-dot" /> Local workspace
          <small>Human judgment. Informed by data.</small>
        </div>
      </aside>
      <div className="main">
        <header>
          <div className="breadcrumb">
            Workspace <ChevronRight size={14} />
            <strong>{page}</strong>
          </div>
          <div className="header-actions">
            {dataset && (
              <select
                aria-label="Active dataset"
                value={active || ""}
                onChange={(e) =>
                  void act("Switching dataset…", () =>
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
              Upload Dataset
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
              onDashboard={() => requestPage("Dashboard")}
            />
          ) : page !== "Settings" ? (
            <DatasetRequiredRoute
              status={datasetStatus}
              hasDataset={!!dataset}
              feature={page}
              onUpload={() => startUpload(page)}
              onCancel={cancelGuard}
              onRetry={() => void act("Checking dataset status…", () => load())}
            >
              <>
                {dataset && (
                  <>
                    <div className="page-heading">
                      <div>
                        <div className="eyebrow">
                          ANNOTATION QUALITY CONTROL
                        </div>
                        <h1>
                          {page === "Dashboard"
                            ? "Dataset overview"
                            : page === "Workload"
                              ? "Review Workload Balancer"
                              : page}
                        </h1>
                        <p>
                          {page === "Smart Sampling"
                            ? "WHAT should we review?"
                            : page === "Workload"
                              ? "WHO reviews it?"
                              : page === "Review Queue"
                                ? "WHAT should be reviewed first?"
                                : "A clear picture of your dataset and the effort ahead."}
                        </p>
                      </div>
                      <div className="dataset-status">
                        <span className="online-dot" /> Analysis complete
                        <small>
                          {dataset.name} · {dataset.format} ·{" "}
                          {title(dataset.task_type)}
                        </small>
                      </div>
                    </div>
                    <div className="principle">
                      <Eye size={16} />
                      Difficulty estimates review effort. It does not predict
                      whether an annotation is correct or incorrect.
                    </div>
                  </>
                )}
                {dataset && page === "Dashboard" && (
                  <>
                    <div className="stats">
                      <Metric
                        label="Total samples"
                        value={dataset.sample_count}
                        note={`${dataset.missing_media} without media`}
                        icon={<Database />}
                      />
                      <Metric
                        label="Annotations"
                        value={dataset.annotation_count}
                        note={`${Object.keys(dataset.classes).length} distinct classes`}
                        icon={<ScanLine />}
                      />
                      <Metric
                        label="Average difficulty"
                        value={fmt(dataset.average_difficulty)}
                        note="Estimated review effort · /100"
                        icon={<ChartNoAxesCombined />}
                      />
                      <Metric
                        label="Hard samples"
                        value={dataset.levels.HARD || 0}
                        note={`${((100 * (dataset.levels.HARD || 0)) / Math.max(1, dataset.sample_count)).toFixed(1)}% of dataset`}
                        icon={<AlertTriangle />}
                      />
                    </div>
                    <div className="grid two">
                      <section className="card">
                        <CardTitle
                          title="Difficulty distribution"
                          sub="Samples by estimated review effort"
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
                          title="Review effort mix"
                          sub="Easy · Medium · Hard · Unavailable"
                        />
                        <Chart
                          pie
                          data={Object.entries(dataset.levels).map(
                            ([name, value]) => ({ name, value }),
                          )}
                        />
                        <div className="legend">
                          {Object.entries(dataset.levels).map(([k, v], i) => (
                            <span key={k}>
                              <i style={{ background: colors[i % 5] }} />
                              {k} <strong>{v}</strong>
                            </span>
                          ))}
                        </div>
                      </section>
                      <section className="card">
                        <CardTitle
                          title="Class distribution"
                          sub="Annotation counts from your imported dataset"
                        />
                        {dataset.annotation_count ? (
                          <Chart
                            data={Object.entries(dataset.classes).map(
                              ([name, value]) => ({ name, value }),
                            )}
                          />
                        ) : (
                          <Empty text="No valid annotations in this dataset." />
                        )}
                      </section>
                      <section className="card">
                        <CardTitle
                          title="Dataset insights"
                          sub="Measured signals, transparent limitations"
                        />
                        <div className="insights">
                          <p>
                            <span>{dataset.rare_classes.length}</span> rare
                            classes at the configured threshold.
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
                            samples with high visual review difficulty.
                          </p>
                          <p>
                            <span>{dataset.missing_media}</span> samples have
                            unavailable media.
                          </p>
                          <p>
                            <span>{dataset.levels.EASY || 0}</span> easy ·{" "}
                            <span>{dataset.levels.MEDIUM || 0}</span> medium ·{" "}
                            <span>{dataset.levels.HARD || 0}</span> hard
                            samples.
                          </p>
                          <p>
                            {Object.entries(dataset.annotation_types)
                              .map(([k, v]) => `${title(k)}: ${v}`)
                              .join(" · ") || "No supported shapes"}
                          </p>
                        </div>
                      </section>
                    </div>
                    <div className="grid two">
                      {dataset.annotation_count > 0 && (
                        <section className="card">
                          <CardTitle
                            title="Annotation types"
                            sub="Normalized shapes across the dataset"
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
                            title="Visual difficulty distribution"
                            sub="Measured images only; missing media is excluded"
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
                          {dataset.warnings.length} import warnings and
                          availability notes
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
                        title="Highest effort samples"
                        sub="A starting point for review — not a prediction of issues"
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
                        void act("Generating QC sample…", async () => {
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
                          title={`${run.selections.length} selected from ${dataset.sample_count} samples`}
                          sub={
                            run.needs_regeneration
                              ? "Needs Regeneration — scores have changed"
                              : "Deterministic selection · no duplicate samples"
                          }
                          action={exportLink("smart_sample")}
                        />
                        {run.needs_regeneration && (
                          <div className="alert">Needs Regeneration</div>
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
                      <Empty text="Set a QC budget and generate your first review subset." />
                    )}
                  </>
                )}
                {dataset && page === "Workload" && (
                  <>
                    <section className="card">
                      <CardTitle
                        title="Your review team"
                        sub="Assignments balance estimated effort across the selected QC subset."
                      />
                      <ReviewerForm
                        disabled={!!busy}
                        onAdd={(name) =>
                          void act("Adding reviewer…", async () => {
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
                                void act("Removing reviewer…", async () => {
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
                          void act("Balancing workload…", async () => {
                            await api(`/datasets/${active}/balance`, {
                              method: "POST",
                            });
                            await load();
                          })
                        }
                      >
                        <Users size={16} />
                        Balance workload
                      </button>
                      {!run && (
                        <p className="muted">
                          Generate a Smart Sampling subset first.
                        </p>
                      )}
                      {run?.needs_regeneration && (
                        <p className="warning-text">
                          QC subset needs regeneration before balancing.
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
                                {unavailable ? "Unavailable" : total.toFixed(1)}
                                <small>Total difficulty</small>
                              </div>
                              <p className="muted">
                                {rows.length} samples · Average{" "}
                                {unavailable
                                  ? "Unavailable"
                                  : rows.length
                                    ? (total / rows.length).toFixed(1)
                                    : "—"}{" "}
                                ·{" "}
                                {rows.filter((s) => s.level === "HARD").length}{" "}
                                hard
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
                            title="Estimated review workload"
                            sub="Sum of assigned sample difficulty"
                          />
                          {assignments.some(
                            (s) => s.overall_difficulty === null,
                          ) ? (
                            <Empty text="Workload unavailable — regenerate this plan after adjusting scoring weights." />
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
                                ? "Needs Regeneration"
                                : "Generate a new subset to rebalance after saving review decisions."
                            }
                            action={exportLink("review_assignments")}
                          />
                          <div className="table-wrap">
                            <table>
                              <thead>
                                <tr>
                                  <th>Sample</th>
                                  <th>Task</th>
                                  <th>Difficulty</th>
                                  <th>Level</th>
                                  <th>Reviewer</th>
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
                        title="Reviewer priority queue"
                        sub="Each reviewer’s samples are ordered by difficulty, highest first."
                        action={exportLink("review_results")}
                      />
                      <label>
                        Reviewer
                        <select
                          value={reviewer || ""}
                          onChange={(e) => setReviewer(Number(e.target.value))}
                        >
                          <option value="" disabled>
                            Select reviewer
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
                          Needs Regeneration — these assignments were based on
                          previous scores.
                        </div>
                      )}
                      <div className="queue-stats">
                        {[
                          ["Total", queue.length],
                          [
                            "Reviewed",
                            queue.filter(
                              (s) =>
                                s.status === "REVIEWED" ||
                                s.status === "ISSUE_FOUND",
                            ).length,
                          ],
                          [
                            "Issues found",
                            queue.filter((s) => s.status === "ISSUE_FOUND")
                              .length,
                          ],
                          [
                            "Pending",
                            queue.filter((s) => s.status === "PENDING").length,
                          ],
                          [
                            "Skipped",
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
                        void act("Recalculating Difficulty…", async () => {
                          await api(`/datasets/${active}/recalculate`, {
                            method: "POST",
                          });
                          await load();
                        })
                      }
                    >
                      <RotateCcw size={15} />
                      Recalculate Difficulty
                    </button>
                    <button
                      className="danger secondary"
                      disabled={!!busy}
                      onClick={() => {
                        if (
                          window.confirm(
                            `Delete ${dataset.name} and its analysis, selections and reviews?`,
                          )
                        )
                          void act("Deleting dataset…", async () => {
                            await api(`/datasets/${active}`, {
                              method: "DELETE",
                            });
                            navigate("Home");
                            await load(null);
                          });
                      }}
                    >
                      <Trash2 size={15} />
                      Delete Dataset
                    </button>
                  </div>
                )}
              </>
            </DatasetRequiredRoute>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">WORKSPACE PREFERENCES</div>
                  <h1>Settings</h1>
                  <p>Customize your interface and scoring preferences.</p>
                </div>
              </div>
              <AppearanceSettings />
              {settings ? (
                <SettingsForm
                  settings={settings}
                  disabled={!!busy}
                  onSave={(s) =>
                    void act("Recalculating Difficulty…", async () => {
                      await api("/settings", json("PUT", s));
                      await load();
                      setNotice(
                        "Settings saved and difficulty recalculated. Existing QC selections and assignments need regeneration.",
                      );
                    })
                  }
                />
              ) : datasetStatus === "error" ? (
                <section className="card" role="alert">
                  <h2>Scoring settings unavailable</h2>
                  <p>
                    Appearance works locally. Reconnect to the backend to load
                    scoring settings.
                  </p>
                  <button
                    onClick={() =>
                      void act("Checking dataset status…", () => load())
                    }
                  >
                    Retry
                  </button>
                </section>
              ) : (
                <div className="busy" role="status">
                  <LoaderCircle className="spin" size={18} />
                  Loading scoring settings…
                </div>
              )}
            </>
          )}
        </main>
        <footer>
          AnnoPilot <span>Intelligent Annotation Review Assistant</span>
          <span>Local · CPU analysis · Human decisions</span>
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
            void act("Uploading and analyzing dataset…", async () => {
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
            void act("Saving review…", async () => {
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
        title={difficulty ? "Difficulty explorer" : "Imported samples"}
        sub={`${rows.length} samples · Click a row to inspect media, annotations and scoring.`}
      />
      <div className="filters">
        <div className="search">
          <Search size={16} />
          <input
            aria-label="Search samples"
            placeholder="Search file name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <select
          aria-label="Difficulty filter"
          value={level}
          onChange={(e) => setLevel(e.target.value)}
        >
          <option value="">All levels</option>
          {["EASY", "MEDIUM", "HARD", "UNAVAILABLE"].map((v) => (
            <option key={v}>{v}</option>
          ))}
        </select>
        <select
          aria-label="Task filter"
          value={task}
          onChange={(e) => setTask(e.target.value)}
        >
          <option value="">All tasks</option>
          {Array.from(new Set(samples.map((s) => s.task_type))).map((v) => (
            <option key={v}>{v}</option>
          ))}
        </select>
        <select
          aria-label="Sort samples"
          value={sort}
          onChange={(e) => setSort(e.target.value)}
        >
          <option value="desc">Difficulty: high first</option>
          <option value="asc">Difficulty: low first</option>
          <option value="name">File name</option>
        </select>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={rare}
            onChange={(e) => setRare(e.target.checked)}
          />
          Rare only
        </label>
      </div>
      <SampleTable
        rows={rows.slice((page - 1) * 20, page * 20)}
        onOpen={onOpen}
      />
      <div className="pagination">
        <span>
          {rows.length
            ? `${(page - 1) * 20 + 1}–${Math.min(page * 20, rows.length)} of ${rows.length}`
            : "0 samples"}
        </span>
        <button
          className="secondary"
          disabled={page === 1}
          onClick={() => setPage(page - 1)}
        >
          Previous
        </button>
        <button
          className="secondary"
          disabled={page * 20 >= rows.length}
          onClick={() => setPage(page + 1)}
        >
          Next
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
        title="Build your QC subset"
        sub="Coverage, effort and edge signals — combined into one review plan."
      />
      <div className="form-grid">
        <label>
          Budget type
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="percentage">Percentage</option>
            <option value="count">Number of samples</option>
          </select>
        </label>
        <label>
          QC budget
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
          Random seed
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
              ? "Random coverage"
              : k === "difficulty"
                ? "High difficulty"
                : "Rare / edge cases"}{" "}
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
          Strategy total: {total}% / 100%
        </span>
        <button disabled={disabled || total !== 100}>
          <Shuffle size={16} />
          Generate QC sample
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
        placeholder="Reviewer name"
        aria-label="Reviewer name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <button disabled={disabled || !name.trim()}>
        <Plus size={16} />
        Add reviewer
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
          <div className="eyebrow">EXPLAINABLE HEURISTICS</div>
          <h2>Scoring settings</h2>
          <p>
            Save to recalculate all datasets and mark existing review plans for
            regeneration.
          </p>
        </div>
        <button disabled={disabled || !valid}>
          <RotateCcw size={16} />
          Save & recalculate
        </button>
      </div>
      <div className="grid two">
        {Object.entries(draft).map(([group, value]) => (
          <section className="card" key={group}>
            <CardTitle
              title={title(group)}
              sub={
                typeof value === "number"
                  ? "Ratio threshold between 0 and 1"
                  : "Weights must sum to 1.00; unavailable features are renormalized."
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
                  Total{" "}
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
            <div className="eyebrow">START WITH YOUR DATA</div>
            <h2 id="upload-title">Upload Dataset</h2>
          </div>
          <button
            aria-label="Close upload"
            className="icon-button"
            disabled={disabled}
            onClick={onClose}
          >
            <X />
          </button>
        </div>
        <p className="muted">
          Import existing annotations and their matching media. Analysis runs
          locally.
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
            Dataset name
            <input
              name="name"
              placeholder="Name your dataset"
              required
              maxLength={200}
            />
          </label>
          <div className="form-grid">
            <label>
              Annotation type
              <select name="task_type">
                <option value="AUTO">Auto Detect</option>
                <option value="BBOX_2D">2D Bounding Box</option>
                <option value="SEGMENTATION">Segmentation</option>
                <option value="KEYPOINT">Keypoint / Pose</option>
                <option value="CUBOID_3D">3D Cuboid</option>
              </select>
            </label>
            <label>
              Annotation format
              <select name="format">
                <option value="AUTO">Auto Detect</option>
                {["CVAT", "COCO", "YOLO", "KITTI"].map((f) => (
                  <option key={f}>{f}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="file-drop">
            <Upload size={24} />
            <strong>Annotation files / ZIP</strong>
            <span>CVAT images XML · COCO JSON · YOLO / KITTI TXT</span>
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
            <strong>Media / sensor data / ZIP</strong>
            <span>JPG · JPEG · PNG · BIN · PCD (sensor files retained)</span>
            <input
              type="file"
              name="media"
              multiple
              accept=".jpg,.jpeg,.png,.bin,.pcd,.zip"
            />
          </label>
          <p className="muted small">
            YOLO requires matching images. KITTI point density and projection
            metrics require calibration and remain unavailable in this MVP.
          </p>
          <button className="full" disabled={disabled}>
            {disabled ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Upload size={16} />
            )}
            Upload & Analyze
          </button>
          {disabled && (
            <p className="muted">
              Uploading, validating, parsing, matching media and calculating
              difficulty. Keep this window open.
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
              SAMPLE #{s.id} · {title(s.task_type)}
            </div>
            <h2 id="detail-title">{s.file_name}</h2>
          </div>
          <button
            aria-label="Close sample"
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
                      ? "Sensor media retained. Inspect cuboid metadata below; interactive point-cloud rendering is not implemented."
                      : "Media unavailable"
                  }
                />
              )}
            </div>
            <div className="principle">
              <Eye size={15} />
              Difficulty estimates effort. Human review determines issues.
            </div>
            <section className="card">
              <CardTitle
                title="Annotation properties"
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
                title="Difficulty breakdown"
                sub="Deterministic scores · 0–100"
              />
              <div className="detail-score">
                <strong>{fmt(s.overall_difficulty)}</strong>
                <Badge level={s.level} />
              </div>
              <div className="score-pair">
                <span>
                  Annotation <strong>{fmt(s.annotation_difficulty)}</strong>
                </span>
                <span>
                  Visual <strong>{fmt(s.visual_difficulty)}</strong>
                </span>
              </div>
              <h4>Feature contributions to overall score</h4>
              {Object.entries(s.analysis?.feature_contributions || {})
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => (
                  <div className="contribution" key={k}>
                    <span>{title(k)}</span>
                    <strong>+{v.toFixed(2)}</strong>
                    <div style={{ width: `${Math.min(v, 100)}%` }} />
                  </div>
                ))}
              <h4>Feature availability & measurements</h4>
              {[
                ...Object.entries(s.analysis?.visual.features || {}).map(
                  ([k, v]) => ["Visual · " + k, v] as const,
                ),
                ...Object.entries(s.analysis?.per_type || {}).flatMap(
                  ([kind, v]) =>
                    Object.entries(v.features).map(
                      ([k, f]) => [kind + " · " + k, f] as const,
                    ),
                ),
              ].map(([k, f]) => (
                <details className="feature" key={k}>
                  <summary>
                    {title(k)}
                    <strong>
                      {f.available
                        ? `${((f.value || 0) * 100).toFixed(1)}%`
                        : "Unavailable"}
                    </strong>
                  </summary>
                  <p>
                    {f.reason_if_unavailable ||
                      "Normalized difficulty heuristic"}
                  </p>
                  {f.raw != null && <pre>{JSON.stringify(f.raw, null, 2)}</pre>}
                </details>
              ))}
              {s.rare_classes.length > 0 && (
                <p>Rare classes: {s.rare_classes.join(", ")}</p>
              )}
            </section>
            {s.assignment_id && (
              <section className="card">
                <CardTitle
                  title="Human review"
                  sub={`${s.reviewer} · ${title(s.status || "PENDING")} · ${title(s.sampling_reason || "")}`}
                />
                <label>
                  Review note
                  <textarea
                    rows={4}
                    maxLength={10000}
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Record your observation…"
                  />
                </label>
                <div className="review-buttons">
                  <button
                    disabled={disabled}
                    onClick={() => onReview("REVIEWED", note)}
                  >
                    <Check size={16} />
                    No Issue
                  </button>
                  <button
                    className="issue"
                    disabled={disabled}
                    onClick={() => onReview("ISSUE_FOUND", note)}
                  >
                    <AlertTriangle size={16} />
                    Issue Found
                  </button>
                  <button
                    className="secondary"
                    disabled={disabled}
                    onClick={() => onReview("SKIPPED", note)}
                  >
                    Skip
                  </button>
                  <button
                    className="secondary"
                    disabled={disabled}
                    onClick={() => onReview(s.status || "PENDING", note)}
                  >
                    Save note
                  </button>
                </div>
                <button
                  className="full secondary"
                  disabled={disabled}
                  onClick={onNext}
                >
                  Next Priority Sample
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
