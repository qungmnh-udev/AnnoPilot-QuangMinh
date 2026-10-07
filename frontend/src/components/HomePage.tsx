import {
  ArrowRight,
  Upload,
  Eye,
  Database,
  ShieldAlert,
  ListFilter,
  Layers,
  Zap,
  TrendingDown,
} from "lucide-react";
import type { Dataset } from "../types";
import type { DatasetStatus } from "./DatasetRequiredRoute";

const modules = [
  [
    ShieldAlert,
    "Model-Assisted QC (N2-04D)",
    "Dùng Pretrained Object Detector (YOLOv8) đối soát với Ground Truth BDD100K.",
  ],
  [
    ListFilter,
    "Prioritized Review Queue",
    "Tự động tính QC Score và đẩy các mẫu có nguy cơ lỗi cao nhất lên đầu hàng đợi.",
  ],
  [
    Layers,
    "Dual-Overlay Visual Diff",
    "Trực quan hóa đa lớp: Nhãn người gán (GT), Model dự đoán, và Vùng bất đồng nghi vấn.",
  ],
  [
    Zap,
    "1-Click Closed-Loop Action",
    "Sửa nhãn và thêm bounding box trực tiếp vào cơ sở dữ liệu chỉ với 1 lần bấm chuột.",
  ],
  [
    TrendingDown,
    "Workload Reduction",
    "Tự động lọc các ảnh sạch có độ đồng thuận cao, cắt giảm 60% – 80% công sức kiểm tra.",
  ],
  [
    Eye,
    "Human-in-the-Loop",
    "Mô hình không thay thế con người; reviewer luôn là người ra quyết định thẩm định cuối cùng.",
  ],
] as const;

export function HomePage({
  dataset,
  status,
  onUpload,
  onDashboard,
  onModelQC,
}: {
  dataset?: Dataset;
  status: DatasetStatus;
  onUpload: () => void;
  onDashboard: () => void;
  onModelQC?: () => void;
}) {
  return (
    <section className="home-page" aria-labelledby="home-title">
      <div className="home-hero">
        <div className="eyebrow">
          ANNOPILOT · MODEL-ASSISTED QUALITY CONTROL (N2-04D)
        </div>
        <h1 id="home-title">
          Bắt vật thể bị sót &amp;
          <br />
          <span>sai class trên BDD100K.</span>
        </h1>
        <p>
          Hệ thống Model-Assisted QC sử dụng Pretrained Detector làm Second Reviewer,
          tự động phát hiện Missing Objects &amp; Wrong Classes, xếp hạng hàng đợi ưu tiên
          và hỗ trợ sửa lỗi 1 chạm.
        </p>
        <div className="home-actions">
          {dataset ? (
            <>
              <button className="large" onClick={onModelQC || onDashboard}>
                <ShieldAlert size={20} /> Open Model QC <ArrowRight size={20} />
              </button>
              <button className="secondary" onClick={onDashboard}>
                <Database size={18} /> View Dataset
              </button>
              <button className="secondary" onClick={onUpload}>
                <Upload size={18} /> Upload Dataset
              </button>
            </>
          ) : (
            <button className="large" onClick={onUpload}>
              <Upload size={20} /> Upload BDD100K Dataset <ArrowRight size={20} />
            </button>
          )}
        </div>
        <p className="home-principle">
          Human-in-the-loop: Bất đồng giữa Pretrained Model và Annotator là bằng chứng gợi ý.
          Reviewer luôn giữ quyền quyết định cuối cùng.
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
          <div className="eyebrow">QUY TRÌNH MODEL-ASSISTED QC</div>
          <h2 id="workflow-title">Quy trình Khép kín (Closed-Loop Workflow)</h2>
          <p>Từ dữ liệu gán nhãn thô đến kết quả đối soát và khắc phục lỗi trực tiếp.</p>
        </div>
        <ol className="home-workflow">
          {[
            "Nạp BDD100K",
            "YOLOv8 Inference",
            "IoU Audit Engine",
            "Hàng đợi Ưu tiên",
            "Visual Diff 2 lớp",
            "1-Click Sửa lỗi",
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
          <div className="eyebrow">CÁC TÍNH NĂNG CỐT LÕI</div>
          <h2 id="modules-title">Bộ công cụ Giữ chất lượng Dữ liệu</h2>
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
        <h2 id="support-title">Tập trung cho Xe Tự hành &amp; BDD100K Taxonomy</h2>
        <div className="home-task-list">
          <span>Car / Van / SUV</span>
          <span>Pedestrian (Người đi bộ)</span>
          <span>Rider (Người lái xe 2 bánh)</span>
          <span>Truck vs Bus Prior</span>
          <span>Traffic Light &amp; Sign</span>
        </div>
        <p>
          Hệ thống áp dụng ma trận nhầm lẫn tiền nghiệm (Driving Confusion Priors)
          đặc thù cho lĩnh vực lái xe tự động để phân biệt các lỗi nghiêm trọng
          như sót người đi bộ từ xa, hoặc gán nhầm xe tải thành xe bus.
        </p>
      </section>
    </section>
  );
}
