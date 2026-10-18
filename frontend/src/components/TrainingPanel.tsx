import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Cpu, GraduationCap, LoaderCircle, Play, Square } from 'lucide-react';
import { api, json } from '../api';
import type { YOLOVariant } from '../types';

type ModelName = YOLOVariant | 'rtdetr_l' | 'fasterrcnn_mobilenet_v3';
type Hardware = { gpu_available: boolean; gpu_name: string | null; gpu_memory_gb: number | null; message: string };
type Preview = { ready: boolean; eligible?: number; golden?: number; empty?: number; missing_media?: number; duplicates?: number; error?: string };
type Job = { id: string; model_name: ModelName; status: string; epoch?: number; epochs?: number;
  message?: string; error?: string; metrics?: Record<string, number>; checkpoint_id?: number;
  config: { train_count: number; validation_count: number; resolved_device: string; epochs: number } };
type Checkpoint = { id: number; model_name: ModelName; job_id: string; active: boolean; created_at: string };
type History = { jobs: Job[]; checkpoints: Checkpoint[] };
const names: Record<ModelName, string> = {
  yolo26n: 'YOLO26n · Nhanh, dễ bắt đầu', yolo26s: 'YOLO26s · Chi tiết hơn',
  rtdetr_l: 'RT-DETR · Transformer', fasterrcnn_mobilenet_v3: 'Faster R-CNN · Hai giai đoạn',
};
const active = new Set(['QUEUED', 'PREPARING', 'TRAINING', 'VALIDATING', 'REGISTERING']);
const statuses: Record<string, string> = { QUEUED: 'Đang chờ', PREPARING: 'Chuẩn bị ảnh', TRAINING: 'Đang học',
  VALIDATING: 'Đang kiểm tra', REGISTERING: 'Đưa vào QC', COMPLETED: 'Hoàn tất', CANCELLED: 'Đã dừng',
  FAILED: 'Chưa thành công', INTERRUPTED: 'Đã gián đoạn' };

export function TrainingPanel({ datasetId, datasetName, onCheckpointChanged }: {
  datasetId: number; datasetName: string; onCheckpointChanged: (model: ModelName) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [history, setHistory] = useState<History>({ jobs: [], checkpoints: [] });
  const [model, setModel] = useState<ModelName>('yolo26n');
  const [epochs, setEpochs] = useState(30);
  const [device, setDevice] = useState('auto');
  const [imageSize, setImageSize] = useState(640);
  const [batch, setBatch] = useState(1);
  const [reviewed, setReviewed] = useState(false);
  const [includeEmpty, setIncludeEmpty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [error, setError] = useState('');
  const observed = useRef<Set<string> | null>(null);
  const callback = useRef(onCheckpointChanged);
  callback.current = onCheckpointChanged;
  const prefix = `/datasets/${datasetId}/training`;
  const running = history.jobs.find(j => active.has(j.status));
  const latest = running || history.jobs[0];

  const refresh = useCallback(async () => {
    const data = await api<History>(prefix);
    const completed = data.jobs.filter(j => j.status === 'COMPLETED');
    const previous = observed.current;
    observed.current = new Set(completed.map(j => j.id));
    setHistory(data);
    if (previous) {
      for (const job of completed) if (!previous.has(job.id)) await callback.current(job.model_name);
    }
  }, [prefix]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try { if (alive) await refresh(); }
      catch (e) { if (alive) setError(e instanceof Error ? e.message : 'Không tải được tiến độ.'); }
      if (alive) timer = setTimeout(poll, 3000);
    }
    void poll();
    void Promise.all([api<Hardware>('/training/hardware'), api<Preview>(`${prefix}/preview`)])
      .then(([machine, data]) => { if (alive) { setHardware(machine); setPreview(data); } })
      .catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; clearTimeout(timer); };
  }, [open, prefix, refresh]);

  async function start(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError(''); setCancelRequested(false);
    try {
      await api(prefix, json('POST', { model_name: model, epochs, device, image_size: imageSize,
        batch_size: batch, labels_reviewed: reviewed, include_verified_empty: includeEmpty }));
      await refresh();
      await callback.current(model);
    } catch (e) { setError(e instanceof Error ? e.message : 'Không bắt đầu được huấn luyện.'); }
    finally { setBusy(false); }
  }
  async function cancel() {
    if (!running) return;
    setBusy(true); setError('');
    try { await api(`${prefix}/${running.id}/cancel`, json('POST', {})); setCancelRequested(true); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không gửi được yêu cầu dừng.'); }
    finally { setBusy(false); }
  }
  async function useCheckpoint(checkpointId: number | null, modelName: ModelName) {
    setBusy(true); setError('');
    try {
      await api(`${prefix}/checkpoint`, json('POST', { model_name: modelName, checkpoint_id: checkpointId }));
      await refresh(); await callback.current(modelName);
    } catch (e) { setError(e instanceof Error ? e.message : 'Không đổi được mô hình kiểm tra nhãn.'); }
    finally { setBusy(false); }
  }

  return <details className="card training-panel" open={open} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary><GraduationCap size={20} /><strong>Huấn luyện mô hình</strong><span className="muted">Học từ nhãn đã kiểm tra</span></summary>
    <div className="training-content">
      <p>Dùng ảnh và nhãn trong <strong>{datasetName}</strong> để mô hình học cách nhận diện dữ liệu của bạn. Khi hoàn tất, bản mô hình vừa học được tự chọn cho lần kiểm tra nhãn hoặc tạo nhãn gợi ý tiếp theo trên bộ dữ liệu này.</p>
      <div className="training-overview">
        <div><span className="training-step">1</span><div><strong>Chuẩn bị nhãn</strong><p>Sửa và đồng bộ từ CVAT trước khi bắt đầu.</p></div></div>
        <div><span className="training-step">2</span><div><strong>Cho mô hình học</strong><p>Ứng dụng tự chia ảnh học và ảnh kiểm tra.</p></div></div>
        <div><span className="training-step">3</span><div><strong>Kiểm tra bằng mô hình mới</strong><p>Chạy lại kiểm tra nhãn để nhận dự đoán mới.</p></div></div>
      </div>
      <div className="training-readiness" role="status">
        <Cpu size={18} /><span>{hardware ? hardware.gpu_available ? `${hardware.gpu_name} · ${hardware.gpu_memory_gb} GB · GPU sẵn sàng` : hardware.message : 'Đang kiểm tra thiết bị…'}</span>
      </div>
      {preview && <p className="muted">{preview.error || `${preview.eligible} ảnh có thể dùng · ${preview.golden} ảnh thuộc bộ chuẩn được giữ riêng · ${preview.empty} ảnh chưa có nhãn được bỏ qua${preview.missing_media ? ` · ${preview.missing_media} ảnh thiếu dữ liệu gốc` : ''}${preview.duplicates ? ` · ${preview.duplicates} ảnh trùng được gộp` : ''}. Cần ít nhất 4 ảnh khác nhau.`}</p>}
      {error && <p role="alert" className="text-warning">{error}</p>}
      {!running && <form onSubmit={start}>
        <fieldset disabled={busy}>
          <div className="training-fields">
            <label>Mô hình muốn huấn luyện<select value={model} onChange={e => setModel(e.target.value as ModelName)}>{Object.entries(names).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
            <label>Mức huấn luyện<select value={[10, 30, 60].includes(epochs) ? epochs : 'custom'} onChange={e => { if (e.target.value !== 'custom') setEpochs(Number(e.target.value)); }}>
              <option value="10">Thử nhanh · 10 vòng</option><option value="30">Thông thường · 30 vòng</option><option value="60">Học lâu hơn · 60 vòng</option>{![10, 30, 60].includes(epochs) && <option value="custom">Tùy chỉnh · {epochs} vòng</option>}
            </select></label>
          </div>
          <p className="muted">Một vòng là một lượt học qua tập ảnh. Học lâu hơn không bảo đảm chính xác hơn. Với máy ít bộ nhớ, bắt đầu bằng YOLO26n.</p>
          <details className="training-options"><summary>Tùy chọn thêm</summary><div className="training-fields">
            <label>Thiết bị<select value={device} onChange={e => setDevice(e.target.value)}><option value="auto">Tự chọn · ưu tiên GPU</option><option value="gpu" disabled={!hardware?.gpu_available}>GPU NVIDIA</option><option value="cpu">CPU</option></select></label>
            <label>Số vòng học<input type="number" min="1" max="300" required value={epochs} onChange={e => setEpochs(Number(e.target.value))} /></label>
            <label>Kích thước ảnh khi học<select value={imageSize} onChange={e => setImageSize(Number(e.target.value))}><option value="320">320 · Ít bộ nhớ</option><option value="640">640 · Cân bằng</option><option value="960">960 · Cần nhiều bộ nhớ</option></select></label>
            <label>Số ảnh mỗi bước học<input type="number" min="1" max="16" required value={batch} onChange={e => setBatch(Number(e.target.value))} /></label>
          </div><label className="training-check"><input type="checkbox" checked={includeEmpty} onChange={e => setIncludeEmpty(e.target.checked)} />Ảnh không có nhãn là ảnh rỗng tôi đã kiểm tra; cho phép dùng để học.</label></details>
          <label className="training-check"><input type="checkbox" required checked={reviewed} onChange={e => setReviewed(e.target.checked)} />Tôi đã kiểm tra nhãn của tập ảnh này và đồng ý dùng chúng để huấn luyện.</label>
          <button className="primary" type="submit" disabled={!reviewed || !preview || !!preview.error || (!preview.ready && !includeEmpty)}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}Bắt đầu huấn luyện</button>
        </fieldset>
      </form>}
      {latest && <div className={`training-progress ${latest.status === 'COMPLETED' ? 'is-complete' : ''}`} aria-live="polite">
        <div className="training-progress-title">{active.has(latest.status) ? <LoaderCircle className="spin" size={18} /> : latest.status === 'COMPLETED' ? <CheckCircle2 size={18} /> : <Cpu size={18} />}<strong>{statuses[latest.status] || latest.status} · {names[latest.model_name]}</strong></div>
        <p>{latest.message || (latest.status === 'INTERRUPTED' ? 'Tác vụ đã gián đoạn. Bạn có thể bắt đầu lượt mới; mô hình cũ vẫn được giữ.' : 'Đang chờ cập nhật…')}</p>
        {latest.status === 'COMPLETED' && <p>{latest.model_name.startsWith('yolo26') ? "Bước tiếp theo: bấm Kiểm tra nhãn bằng AI để dùng mô hình vừa học." : "Bước tiếp theo: mở Tạo nhãn gợi ý (WBF), rồi bấm Nhận diện và tạo nhãn gợi ý để dùng mô hình vừa học."}</p>}
        <progress max={latest.epochs || latest.config.epochs} value={latest.epoch || 0} aria-label="Tiến độ huấn luyện" />
        <div className="training-progress-meta"><span>{latest.epoch || 0}/{latest.epochs || latest.config.epochs} vòng</span><span>{latest.config.train_count} ảnh học · {latest.config.validation_count} ảnh kiểm tra</span><span>{latest.config.resolved_device.startsWith('cuda') ? 'GPU' : 'CPU'}</span></div>
        {latest.error && <p role="alert" className="text-warning">{latest.error}</p>}
        {running && <button type="button" className="secondary" disabled={busy || cancelRequested} onClick={() => void cancel()}><Square size={14} />{cancelRequested ? 'Đang chờ dừng ở bước an toàn…' : 'Dừng huấn luyện'}</button>}
        <details><summary>Chi tiết kỹ thuật</summary><pre>{JSON.stringify(latest.metrics || {}, null, 2)}</pre><a href={`/api${prefix}/${latest.id}/log`}>Tải nhật ký huấn luyện</a></details>
      </div>}
      {!!history.checkpoints.length && <details className="training-options"><summary>Mô hình đang dùng và các phiên bản trước</summary><ul className="training-checkpoints">{history.checkpoints.map(c => <li key={c.id}>
        <span><strong>{names[c.model_name]}</strong> · #{c.id} · {new Date(c.created_at).toLocaleString('vi-VN')}{c.active && <span className="badge">Đang được chọn để kiểm tra nhãn</span>}</span>
        <button className="secondary" type="button" disabled={busy || !!running} onClick={() => void useCheckpoint(c.active ? null : c.id, c.model_name)}>{c.active ? "Dùng lại mô hình gốc" : 'Dùng phiên bản này'}</button>
      </li>)}</ul></details>}
      <p className="muted">Bạn có thể đóng mục này khi đang huấn luyện; máy chủ vẫn tiếp tục chạy và lưu tiến độ. Bộ nhãn chuẩn Golden được giữ riêng để đánh giá, không dùng để học.</p>
    </div>
  </details>;
}
