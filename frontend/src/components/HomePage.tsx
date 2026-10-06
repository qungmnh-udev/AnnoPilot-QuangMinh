import {
  ArrowRight,
  Upload,
  ChartNoAxesCombined,
  Shuffle,
  Users,
  ListChecks,
  Eye,
  Database,
  ShieldAlert,
} from "lucide-react";
import type { Dataset } from "../types";
import type { DatasetStatus } from "./DatasetRequiredRoute";

const modules = [
  [
    ShieldAlert,
    "Model-Assisted QC (N2-04D)",
    "Bắt vật thể bị sót và sai class bằng Pretrained Detector.",
  ],
  [
    ChartNoAxesCombined,
    "Difficulty Analysis",
    "Estimate how much review effort each sample may require.",
  ],
  [
    Shuffle,
    "Smart Sampling",
    "Select which samples should enter the QC subset.",
  ],
  [
    Users,
    "Workload Balancer",
    "Distribute selected samples based on estimated review effort.",
  ],
  [ListChecks, "Review Priority", "Review difficult assigned samples first."],
  [
    Eye,
    "Human Review",
    "The reviewer makes the final Issue / No Issue decision.",
  ],
] as const;

export function HomePage({
  dataset,
  status,
  onUpload,
  onDashboard,
}: {
  dataset?: Dataset;
  status: DatasetStatus;
  onUpload: () => void;
  onDashboard: () => void;
}) {
  return (
    <section className="home-page" aria-labelledby="home-title">
      <div className="home-hero">
        <div className="eyebrow">
          ANNOPILOT · INTELLIGENT ANNOTATION REVIEW ASSISTANT
        </div>
        <h1 id="home-title">
          Review smarter,
          <br />
          <span>not more.</span>
        </h1>
        <p>
          Analyze annotation complexity, select the right QC samples, balance
          reviewer workload, and prioritize difficult cases.
        </p>
        <div className="home-actions">
          {dataset ? (
            <>
              <button className="large" onClick={onDashboard}>
                Open Dashboard <ArrowRight size={20} />
              </button>
              <button className="secondary" onClick={onUpload}>
                <Upload size={18} /> Upload Dataset
              </button>
            </>
          ) : (
            <button className="large" onClick={onUpload}>
              <Upload size={20} /> Upload Dataset <ArrowRight size={20} />
            </button>
          )}
        </div>
        <p className="home-principle">
          Difficulty estimates review effort, never error probability. People
          decide whether an issue exists.
        </p>
      </div>

      {dataset && (
        <section
          className="card home-dataset"
          aria-labelledby="current-dataset-title"
        >
          <div className="card-title">
            <div>
              <div className="eyebrow">CURRENT DATASET</div>
              <h2 id="current-dataset-title">{dataset.name}</h2>
            </div>
            <Database size={26} />
          </div>
          <dl>
            <div>
              <dt>Task Type</dt>
              <dd>{dataset.task_type.replaceAll("_", " ")}</dd>
            </div>
            <div>
              <dt>Samples</dt>
              <dd>{dataset.sample_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Annotations</dt>
              <dd>{dataset.annotation_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Analysis Status</dt>
              <dd>
                {status === "loading"
                  ? "Refreshing workspace…"
                  : status === "error"
                    ? "Connection unavailable"
                    : "Analysis complete"}
              </dd>
            </div>
          </dl>
          <button className="secondary" onClick={onDashboard}>
            Open Dashboard <ArrowRight size={18} />
          </button>
        </section>
      )}

      <section className="home-section" aria-labelledby="workflow-title">
        <div className="home-section-heading">
          <div className="eyebrow">FROM ANNOTATIONS TO ACTION</div>
          <h2 id="workflow-title">A focused review workflow</h2>
          <p>One connected path from your data to human findings.</p>
        </div>
        <ol className="home-workflow">
          {[
            "Upload",
            "Analyze",
            "Difficulty",
            "Smart Sampling",
            "Balance",
            "Prioritize",
            "Human Review",
          ].map((step, index) => (
            <li key={step}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{step}</strong>
              <ArrowRight size={16} aria-hidden="true" />
            </li>
          ))}
        </ol>
      </section>

      <section className="home-section" aria-labelledby="modules-title">
        <div className="home-section-heading">
          <div className="eyebrow">YOUR REVIEW TOOLKIT</div>
          <h2 id="modules-title">Plan effort. Capture judgment.</h2>
        </div>
        <div className="home-modules">
          {modules.map(([Icon, name, description]) => (
            <article className="card" key={name}>
              <Icon size={26} aria-hidden="true" />
              <h3>{name}</h3>
              <p>{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="card home-support" aria-labelledby="support-title">
        <h2 id="support-title">Built for multiple annotation tasks</h2>
        <div className="home-task-list">
          <span>2D Bounding Box</span>
          <span>Polygon Segmentation</span>
          <span>Keypoint / Pose</span>
          <span>3D Cuboid · metadata review</span>
        </div>
        <p>
          Import CVAT images XML, COCO, YOLO detection or KITTI labels. RLE
          masks are retained with limited analysis; articulated pose skeletons
          and calibrated point-cloud analysis / 3D viewing are not available.
        </p>
      </section>
    </section>
  );
}
