import { useEffect, useState } from 'react';
import { api, json } from '../api';
import { displayLabel } from '../uiText';

type Metrics = { tp: number; fp: number; fn: number; precision: number | null; recall: number | null; f1: number | null };
type Run = { id: number; benchmark_id: number; strategy: string; metrics: {
  MISSING_OBJECT: Metrics; WRONG_CLASS: Metrics; overall: Metrics;
  frame_count: number; qc_seconds: number; inference_seconds: number | null; true_errors_per_review_hour: number | null;
}; created_at: string };
type EvaluationData = { benchmarks: { id: number; verified_by: string; frame_count: number }[]; runs: Run[] };
type Acceptance = { status: string; strategy: string | null; submission_count: number; return_count: number;
  events: { id: number; cycle: number; status: string; actor: string; reason: string; created_at: string }[] };
const percent = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;

export function QCEvaluationPanel({ datasetId }: { datasetId: number }) {
  const [data, setData] = useState<EvaluationData>({ benchmarks: [], runs: [] });
  const [acceptance, setAcceptance] = useState<Acceptance | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [baseline, setBaseline] = useState('');
  const [pilot, setPilot] = useState<string[]>([]);
  const [benchmarkId, setBenchmarkId] = useState('');
  const [minutes, setMinutes] = useState({ baseline: '', multi_model: '' });
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');
  const [strategy, setStrategy] = useState('multi_model');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const prefix = `/datasets/${datasetId}/qc`;

  async function refresh() {
    const [evaluations, history, sources] = await Promise.all([
      api<EvaluationData>(`${prefix}/evaluations`), api<Acceptance>(`${prefix}/acceptance`),
      api<{ models: string[]; active_models?: string[] }>(`/datasets/${datasetId}/ensemble/models`),
    ]);
    setData(evaluations); setAcceptance(history);
    const available = sources.models.filter(m => !m.startsWith('ensemble_') && !m.startsWith('synthetic') && !m.toLowerCase().startsWith('yolov8'));
    const active = (sources.active_models || available).filter(m => available.includes(m));
    setModels(available);
    setBaseline(current => available.includes(current) ? current : (active.find(m => m.includes('yolo')) || available[0] || ''));
    setPilot(current => current.length ? current.filter(m => available.includes(m)) : active);
    setBenchmarkId(current => evaluations.benchmarks.some(b => String(b.id) === current) ? current : String(evaluations.benchmarks.at(-1)?.id || ''));
    if (history.strategy) setStrategy(history.strategy);
  }
  useEffect(() => { void refresh().catch(e => setError(e.message)); }, [datasetId]);
  async function perform(action: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await action(); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không thể hoàn tất thao tác'); }
    finally { setBusy(false); }
  }
  async function importGolden(file: File) {
    await perform(async () => {
      let payload: unknown;
      try { payload = JSON.parse(await file.text()); }
      catch { throw new Error('Không đọc được tệp JSON. Hãy kiểm tra định dạng theo mẫu Bộ nhãn chuẩn bên dưới.'); }
      await api(`${prefix}/benchmarks`, json('POST', payload));
    });
  }
  function compare() {
    void perform(() => api(`${prefix}/evaluations/compare`, json('POST', {
      benchmark_id: Number(benchmarkId), baseline_model: baseline, pilot_models: pilot,
      review_minutes: Object.fromEntries(Object.entries(minutes).filter(([, v]) => v.trim()).map(([k, v]) => [k, Number(v)])),
    })));
  }
  function record(status: string) {
    void perform(() => api(`${prefix}/acceptance`, json('POST', { status, strategy, actor, reason })));
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...data, acceptance }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `qc-evaluation-${datasetId}.json`; link.click(); URL.revokeObjectURL(url);
  }
  return <details className="card qc-evaluation-panel">
    <summary><strong>Đánh giá chất lượng và nghiệm thu</strong></summary>
    <p className="muted">So sánh cách kiểm tra bằng một mô hình (Baseline) với nhiều mô hình (Pilot) trên cùng bộ nhãn chuẩn đã được người kiểm tra xác nhận. Mục này dùng để đo chất lượng, không tự tạo nhãn hoặc huấn luyện mô hình.</p>
    <ol className="plain-steps"><li>Chuẩn bị bộ nhãn chuẩn độc lập, đã được kiểm tra thủ công, rồi nhập tệp JSON bên dưới.</li><li>Chạy nhận diện trước để có dự đoán của ít nhất hai mô hình. Chọn một mô hình đối chiếu và các mô hình kiểm tra chéo, rồi bấm So sánh và lưu kết quả.</li><li>Sau khi người kiểm tra xem xét chất lượng, ghi nhận nộp, yêu cầu sửa hoặc chấp nhận ở phần Nghiệm thu.</li></ol>
    <details className="advanced-options">
      <summary>Bộ nhãn chuẩn là gì? Chuẩn bị tệp nhập như thế nào?</summary>
      <p>Bộ nhãn chuẩn (Golden) là một nhóm ảnh được người kiểm tra rà soát đầy đủ, kể cả các vùng AI không cảnh báo. Giữ riêng nhóm này để đo chất lượng, không dùng để huấn luyện. Tệp nhập dùng định dạng Golden của AnnoPilot, không phải tệp xuất CVAT hoặc COCO nguyên bản.</p>
      <p>Thay tên ảnh, tên loại vật thể và tọa độ bằng dữ liệu đã xác nhận của bạn theo mẫu dưới đây. Tên ảnh phải khớp bộ dữ liệu đang chọn; tọa độ tính bằng điểm ảnh. Ảnh đã xác nhận không có vật thể dùng danh sách annotations rỗng.</p>
      <pre className="golden-format-example">{JSON.stringify({ verified_by: 'Tên người kiểm tra', notes: 'Ghi chú cách kiểm chứng', includes_unwarned_regions: true, samples: [{ file_name: 'ten-anh-trong-bo-du-lieu.jpg', annotations: [{ label: 'car', geometry: { x1: 10, y1: 20, x2: 150, y2: 200 } }] }] }, null, 2)}</pre>
      <p className="muted">Chỉ đặt includes_unwarned_regions thành true sau khi đã rà soát toàn ảnh. Nhãn chuẩn cần được chuẩn bị trước khi sửa nhãn đầu vào nếu muốn đo khả năng tìm lỗi.</p>
    </details>
    {error && <p role="alert" className="text-warning">{error}</p>}
    <fieldset disabled={busy} style={{ border: 0, padding: 0 }}>
      <div className="qc-evaluation-controls">
        <label>Nhập bộ nhãn chuẩn (Golden, tệp JSON)<input type="file" accept=".json,application/json" onChange={e => { const file = e.target.files?.[0]; if (file) void importGolden(file); e.target.value = ''; }} /></label>
        <label>Tập kiểm chứng<select value={benchmarkId} onChange={e => setBenchmarkId(e.target.value)}><option value="">Chưa có tập kiểm chứng</option>{data.benchmarks.map(b => <option key={b.id} value={b.id}>#{b.id} · {b.frame_count} ảnh · {b.verified_by}</option>)}</select></label>
        <label>Mô hình đối chiếu ban đầu<select value={baseline} onChange={e => { setBaseline(e.target.value); setPilot(current => [...new Set([...current, e.target.value])]); }}><option value="">Chọn mô hình</option>{models.map(m => <option key={m}>{m}</option>)}</select></label>
      </div>
      <p>Các mô hình dùng để kiểm tra chéo:</p>
      <div className="qc-evaluation-controls">{models.map(m => <label key={m}><input type="checkbox" checked={pilot.includes(m)} disabled={m === baseline} onChange={e => setPilot(current => e.target.checked ? [...current, m] : current.filter(x => x !== m))} /> {m}</label>)}</div>
      <div className="qc-evaluation-controls">
        <label>Thời gian kiểm tra bằng một mô hình (phút, tùy chọn)<input type="number" min="0.01" step="any" value={minutes.baseline} onChange={e => setMinutes({ ...minutes, baseline: e.target.value })} /></label>
        <label>Thời gian kiểm tra bằng nhiều mô hình (phút, tùy chọn)<input type="number" min="0.01" step="any" value={minutes.multi_model} onChange={e => setMinutes({ ...minutes, multi_model: e.target.value })} /></label>
        <button className="primary" disabled={!benchmarkId || !baseline || pilot.length < 2} onClick={compare}>So sánh và lưu kết quả</button>
        <button onClick={download}>Tải báo cáo JSON</button>
        <button onClick={() => void perform(refresh)}>Tải lại danh sách mô hình</button>
      </div>
      <p className="muted">Cần bộ nhãn chuẩn và ít nhất hai nguồn dự đoán để so sánh. Tỷ lệ tìm được lỗi cho biết đã phát hiện bao nhiêu lỗi thật; tỷ lệ cảnh báo đúng cho biết bao nhiêu cảnh báo là chính xác. Tên nguồn khác nhau chưa bảo đảm đó là các mô hình độc lập.</p>
      <p className="muted">TP / FP / FN lần lượt là số lỗi tìm đúng / cảnh báo sai / lỗi bị bỏ sót. Thời gian nhận diện và đối chiếu được tính riêng; thời gian kiểm tra thủ công là số phút bạn tự ghi nhận.</p>
      <div style={{ overflowX: 'auto' }}><table><thead><tr><th>Lượt đo / Bộ chuẩn</th><th>Cấu hình</th><th>Tỷ lệ tìm được lỗi thiếu vật thể</th><th>Tỷ lệ tìm được lỗi sai loại</th><th>Tỷ lệ cảnh báo đúng</th><th>TP / FP / FN</th><th>Lỗi thật / giờ kiểm tra</th><th>Nhận diện / Đối chiếu (giây)</th></tr></thead><tbody>
        {data.runs.map(r => <tr key={r.id}><td>#{r.id} / #{r.benchmark_id}</td><td>{r.strategy === 'baseline' ? "Một mô hình" : "Nhiều mô hình"} · {r.metrics.frame_count} ảnh</td><td>{percent(r.metrics.MISSING_OBJECT.recall)}</td><td>{percent(r.metrics.WRONG_CLASS.recall)}</td><td>{percent(r.metrics.overall.precision)}</td><td>{r.metrics.overall.tp} / {r.metrics.overall.fp} / {r.metrics.overall.fn}</td><td>{r.metrics.true_errors_per_review_hour?.toFixed(1) ?? '—'}</td><td>{r.metrics.inference_seconds?.toFixed(3) ?? '—'} / {r.metrics.qc_seconds.toFixed(3)}</td></tr>)}
        {!data.runs.length && <tr><td colSpan={8}>Chưa có kết quả đo. Dấu — nghĩa là chưa có dữ liệu để tính hoặc chưa ghi thời gian kiểm tra.</td></tr>}
      </tbody></table></div>
      <p><strong>Nghiệm thu:</strong> {acceptance ? displayLabel(acceptance.status) : 'Đang tải'} · {acceptance?.submission_count || 0} lần nộp · {acceptance?.return_count || 0} vòng bị trả lại</p>
      <div className="qc-evaluation-controls">
        <label>Quy trình<select value={strategy} disabled={!!acceptance?.strategy} onChange={e => setStrategy(e.target.value)}><option value="baseline">Một mô hình</option><option value="multi_model">Nhiều mô hình</option></select></label>
        <label>Người ghi nhận<input value={actor} onChange={e => setActor(e.target.value)} /></label>
        <label>Lý do / ghi chú<input value={reason} onChange={e => setReason(e.target.value)} /></label>
        <button disabled={!actor.trim() || !['NOT_SUBMITTED', 'RETURNED'].includes(acceptance?.status || '')} onClick={() => record('SUBMITTED')}>Ghi nhận nộp nghiệm thu</button>
        <button disabled={!actor.trim() || !reason.trim() || acceptance?.status !== 'SUBMITTED'} onClick={() => record('RETURNED')}>Ghi nhận trả lại</button>
        <button disabled={!actor.trim() || acceptance?.status !== 'SUBMITTED'} onClick={() => record('ACCEPTED')}>Ghi nhận chấp nhận</button>
      </div>
      <p className="muted">Đây là nhật ký quyết định của người dùng, không tự chấm đạt. Nhập tên người ghi nhận để nộp; khi yêu cầu sửa lại cần có lý do. Chỉ chấp nhận hoặc trả lại sau khi đã nộp.</p>
      {acceptance?.events.map(e => <p key={e.id}>Lần nộp {e.cycle} · {displayLabel(e.status)} · {e.actor} · {e.reason || 'Không có ghi chú'}</p>)}
    </fieldset>
  </details>;
}
