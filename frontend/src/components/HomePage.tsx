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
import { displayLabel } from "../uiText";

const modules = [
  [
    ShieldAlert,
    "Kiểm tra nhãn bằng AI",
    "Đối chiếu nhãn với dự đoán để tìm vị trí có thể thiếu vật thể hoặc gán sai loại.",
  ],
  [
    ListFilter,
    "Danh sách ưu tiên kiểm tra",
    "Ảnh có nhiều nghi vấn được xếp trước để bạn biết nên bắt đầu từ đâu.",
  ],
  [
    Layers,
    "So sánh ngay trên ảnh",
    "Bật hoặc tắt nhãn hiện có, dự đoán AI và vùng nghi vấn để dễ đối chiếu.",
  ],
  [
    Zap,
    "Xử lý từng nghi vấn",
    "Chấp nhận gợi ý trong AnnoPilot hoặc mở CVAT để sửa nhãn gốc, rồi đồng bộ lại.",
  ],
  [
    TrendingDown,
    "Đo chất lượng trước và sau",
    "Dùng bộ nhãn chuẩn để so sánh một mô hình với nhiều mô hình, rồi ghi nhận nghiệm thu.",
  ],
  [
    Eye,
    "Bạn là người quyết định",
    "AI đưa ra gợi ý. Bạn kiểm tra ảnh và xác nhận kết quả cuối cùng.",
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
          ANNOPILOT · TRỢ LÝ KIỂM TRA NHÃN
        </div>
        <h1 id="home-title">
          Tìm vật thể bị sót và
          <br />
          <span>nhãn sai loại trong ảnh.</span>
        </h1>
        <p>
          Thêm ảnh, để AI gợi ý những chỗ cần xem lại, rồi kiểm tra và sửa nhãn. Nếu ảnh chưa có nhãn, bạn có thể tạo nhãn gợi ý bằng nhiều mô hình để bắt đầu nhanh hơn.
        </p>
        <div className="home-actions">
          {dataset ? (
            <>
              <button className="large" onClick={onModelQC || onDashboard}>
                <ShieldAlert size={20} /> Bắt đầu kiểm tra nhãn <ArrowRight size={20} />
              </button>
              <button className="secondary" onClick={onDashboard}>
                <Database size={18} /> Xem bộ dữ liệu
              </button>
              <button className="secondary" onClick={onUpload}>
                <Upload size={18} /> Thêm bộ dữ liệu
              </button>
            </>
          ) : (
            <button className="large" onClick={onUpload}>
              <Upload size={20} /> Thêm bộ dữ liệu đầu tiên <ArrowRight size={20} />
            </button>
          )}
        </div>
        <p className="home-principle">
          Cảnh báo của AI là gợi ý để bạn kiểm tra, chưa phải kết luận nhãn sai.
        </p>
      </div>

      {dataset && (
        <section
          className="card home-dataset"
          aria-labelledby="current-dataset-title"
        >
          <div className="card-title">
            <div>
              <div className="eyebrow">BỘ DỮ LIỆU ĐANG CHỌN</div>
              <h2 id="current-dataset-title">{dataset.name}</h2>
            </div>
            <Database size={26} />
          </div>
          <dl>
            <div>
              <dt>Dạng nhãn</dt>
              <dd>{displayLabel(dataset.task_type)}</dd>
            </div>
            <div>
              <dt>Số mẫu</dt>
              <dd>{dataset.sample_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Nhãn</dt>
              <dd>{dataset.annotation_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Trạng thái phân tích</dt>
              <dd>
                {status === "loading"
                  ? "Đang cập nhật dữ liệu…"
                  : status === "error"
                    ? "Chưa kết nối được máy chủ"
                    : "Đã phân tích"}
              </dd>
            </div>
          </dl>
          <button className="secondary" onClick={onDashboard}>
            Xem bộ dữ liệu <ArrowRight size={18} />
          </button>
        </section>
      )}

      <section className="home-section" aria-labelledby="workflow-title">
        <div className="home-section-heading">
          <div className="eyebrow">CÁCH SỬ DỤNG</div>
          <h2 id="workflow-title">Bắt đầu từ ảnh, hoàn thiện bằng kiểm tra</h2>
          <p>Đi theo các bước dưới đây. Bạn có thể dùng mô hình có sẵn, chưa cần huấn luyện mô hình mới.</p>
        </div>
        <ol className="home-workflow">
          {[
            "Thêm ảnh và nhãn",
            "Cho AI nhận diện",
            "Đối chiếu với nhãn",
            "Xem ảnh có nghi vấn",
            "So sánh trên ảnh",
            "Sửa và đồng bộ nhãn",
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
          <h2 id="modules-title">Mỗi chức năng giúp bạn làm gì?</h2>
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
        <h2 id="support-title">Hỗ trợ các loại vật thể giao thông</h2>
        <div className="home-task-list">
          <span>Ô tô / xe van / SUV</span>
          <span>Người đi bộ</span>
          <span>Người đi xe hai bánh</span>
          <span>Xe tải và xe buýt</span>
          <span>Đèn và biển báo giao thông</span>
        </div>
        <p>
          Ứng dụng ưu tiên các tình huống dễ nhầm trong ảnh giao thông, ví dụ bỏ sót người đi bộ hoặc nhầm xe tải với xe buýt. Tên loại vật thể trong tệp dữ liệu vẫn được giữ nguyên để trao đổi đúng với CVAT.
        </p>
      </section>
    </section>
  );
}
