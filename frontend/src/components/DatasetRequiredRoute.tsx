import { useEffect, useRef, type ReactNode } from "react";
import { AlertTriangle, Database, LoaderCircle, Upload, X } from "lucide-react";
import { displayLabel } from "../uiText";

export type DatasetStatus = "loading" | "ready" | "error";
export const featureLabel = displayLabel;

export function DatasetRequiredState({
  feature,
  onUpload,
  onCancel,
}: {
  feature: string;
  onUpload: () => void;
  onCancel: () => void;
}) {
  return (
    <section className="card dataset-required-state">
      <Database size={32} />
      <h1>Cần thêm bộ dữ liệu</h1>
      <p>Thêm bộ dữ liệu để sử dụng mục {featureLabel(feature)}.</p>
      <p className="muted">
        Hãy thêm ảnh và nhãn của bạn để bắt đầu. Bạn có thể tạo nhãn gợi ý nếu ảnh chưa có nhãn.
      </p>
      <div className="guard-actions">
        <button className="secondary" onClick={onCancel}>
          Ở lại trang này
        </button>
        <button onClick={onUpload}>
          <Upload size={16} />
          Thêm bộ dữ liệu
        </button>
      </div>
    </section>
  );
}

export function DatasetRequiredRoute({
  status,
  hasDataset,
  feature,
  onUpload,
  onCancel,
  onRetry,
  children,
}: {
  status: DatasetStatus;
  hasDataset: boolean;
  feature: string;
  onUpload: () => void;
  onCancel: () => void;
  onRetry: () => void;
  children: ReactNode;
}) {
  if (status === "loading")
    return (
      <section className="card guard-status" role="status">
        <LoaderCircle className="spin" size={24} />
        <h2>Đang kiểm tra bộ dữ liệu…</h2>
        <p>Vui lòng chờ ứng dụng tải dữ liệu của bạn.</p>
      </section>
    );
  if (status === "error")
    return (
      <section className="card guard-status" role="alert">
        <AlertTriangle size={24} />
        <h2>Không kiểm tra được trạng thái bộ dữ liệu.</h2>
        <p>
          Hãy kiểm tra kết nối máy chủ rồi thử lại. Dữ liệu đã lưu có thể vẫn còn trên máy chủ.
        </p>
        <button onClick={onRetry}>Thử lại</button>
      </section>
    );
  if (!hasDataset)
    return (
      <DatasetRequiredState
        feature={feature}
        onUpload={onUpload}
        onCancel={onCancel}
      />
    );
  return children;
}

export function DatasetRequiredDialog({
  feature,
  onUpload,
  onCancel,
}: {
  feature: string;
  onUpload: () => void;
  onCancel: () => void;
}) {
  const dialog = useRef<HTMLElement>(null);
  const cancel = useRef(onCancel);
  cancel.current = onCancel;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const root = dialog.current;
    root?.querySelector<HTMLButtonElement>("[data-cancel]")?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        cancel.current();
      }
      if (event.key === "Tab" && root) {
        const buttons = Array.from(
          root.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"),
        );
        const first = buttons[0],
          last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  return (
    <div className="modal-backdrop">
      <section
        ref={dialog}
        className="modal dataset-required-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-required-title"
        aria-describedby="dataset-required-description"
      >
        <div className="modal-heading">
          <div className="guard-heading">
            <Database size={24} />
            <h2 id="dataset-required-title">Cần thêm bộ dữ liệu</h2>
          </div>
          <button
            aria-label="Đóng thông báo cần bộ dữ liệu"
            className="icon-button"
            onClick={onCancel}
          >
            <X size={20} />
          </button>
        </div>
        <p id="dataset-required-description">
          Thêm bộ dữ liệu để sử dụng mục {featureLabel(feature)}.
        </p>
        <p className="muted">
          Hãy thêm ảnh và nhãn của bạn để bắt đầu. Bạn có thể tạo nhãn gợi ý nếu ảnh chưa có nhãn.
        </p>
        <div className="guard-actions">
          <button className="secondary" data-cancel onClick={onCancel}>
            Ở lại trang này
          </button>
          <button onClick={onUpload}>
            <Upload size={16} />
            Thêm bộ dữ liệu
          </button>
        </div>
      </section>
    </div>
  );
}
