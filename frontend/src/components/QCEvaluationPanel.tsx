import { useEffect, useState } from 'react';
import { api, json } from '../api';

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
      const payload = JSON.parse(await file.text());
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
    <summary><strong>Đánh giá Baseline → Pilot & vòng nghiệm thu</strong></summary>
    <p className="muted">So sánh một detector với kiểm tra chéo nhiều detector trên cùng nhãn đầu vào và tập kiểm chứng độc lập. Hãy đánh giá trước khi sửa nhãn; ảnh không có cảnh báo cũng cần được kiểm chứng.</p>
    {error && <p role="alert" className="text-warning">{error}</p>}
    <fieldset disabled={busy} style={{ border: 0, padding: 0 }}>
      <div className="qc-evaluation-controls">
        <label>Golden Benchmark JSON<input type="file" accept=".json,application/json" onChange={e => { const file = e.target.files?.[0]; if (file) void importGolden(file); e.target.value = ''; }} /></label>
        <label>Tập kiểm chứng<select value={benchmarkId} onChange={e => setBenchmarkId(e.target.value)}><option value="">Chưa có tập kiểm chứng</option>{data.benchmarks.map(b => <option key={b.id} value={b.id}>#{b.id} · {b.frame_count} frame · {b.verified_by}</option>)}</select></label>
        <label>Detector baseline<select value={baseline} onChange={e => { setBaseline(e.target.value); setPilot(current => [...new Set([...current, e.target.value])]); }}><option value="">Chọn detector</option>{models.map(m => <option key={m}>{m}</option>)}</select></label>
      </div>
      <p>Detector pilot:</p>
      <div className="qc-evaluation-controls">{models.map(m => <label key={m}><input type="checkbox" checked={pilot.includes(m)} disabled={m === baseline} onChange={e => setPilot(current => e.target.checked ? [...current, m] : current.filter(x => x !== m))} /> {m}</label>)}</div>
      <div className="qc-evaluation-controls">
        <label>Review baseline (phút, nếu đã đo)<input type="number" min="0.01" step="any" value={minutes.baseline} onChange={e => setMinutes({ ...minutes, baseline: e.target.value })} /></label>
        <label>Review pilot (phút, nếu đã đo)<input type="number" min="0.01" step="any" value={minutes.multi_model} onChange={e => setMinutes({ ...minutes, multi_model: e.target.value })} /></label>
        <button className="primary" disabled={!benchmarkId || !baseline || pilot.length < 2} onClick={compare}>So sánh và lưu kết quả</button>
        <button onClick={download}>Tải báo cáo JSON</button>
        <button onClick={() => void perform(refresh)}>Cập nhật nguồn model</button>
      </div>
      <p className="muted">Tên nguồn khác nhau chưa chứng minh checkpoint độc lập. Recall và precision chỉ có khi có tập kiểm chứng; thời gian QC dưới đây không bao gồm inference.</p>
      <div style={{ overflowX: 'auto' }}><table><thead><tr><th>Run / Benchmark</th><th>Cấu hình</th><th>Recall sót vật thể</th><th>Recall sai class</th><th>Precision tổng</th><th>TP / FP / FN</th><th>Lỗi thật/giờ review</th><th>Inference / QC (giây)</th></tr></thead><tbody>
        {data.runs.map(r => <tr key={r.id}><td>#{r.id} / #{r.benchmark_id}</td><td>{r.strategy === 'baseline' ? 'Baseline' : 'Pilot'} · {r.metrics.frame_count} frame</td><td>{percent(r.metrics.MISSING_OBJECT.recall)}</td><td>{percent(r.metrics.WRONG_CLASS.recall)}</td><td>{percent(r.metrics.overall.precision)}</td><td>{r.metrics.overall.tp} / {r.metrics.overall.fp} / {r.metrics.overall.fn}</td><td>{r.metrics.true_errors_per_review_hour?.toFixed(1) ?? '—'}</td><td>{r.metrics.inference_seconds?.toFixed(3) ?? '—'} / {r.metrics.qc_seconds.toFixed(3)}</td></tr>)}
        {!data.runs.length && <tr><td colSpan={8}>Chưa có kết quả đo. Dấu — nghĩa là chưa có mẫu hoặc chưa ghi thời gian review.</td></tr>}
      </tbody></table></div>
      <p><strong>Nghiệm thu:</strong> {acceptance?.status || 'Đang tải'} · {acceptance?.submission_count || 0} lần nộp · {acceptance?.return_count || 0} vòng bị trả lại</p>
      <div className="qc-evaluation-controls">
        <label>Quy trình<select value={strategy} disabled={!!acceptance?.strategy} onChange={e => setStrategy(e.target.value)}><option value="baseline">Baseline</option><option value="multi_model">Pilot multi-model</option></select></label>
        <label>Người ghi nhận<input value={actor} onChange={e => setActor(e.target.value)} /></label>
        <label>Lý do / ghi chú<input value={reason} onChange={e => setReason(e.target.value)} /></label>
        <button disabled={!actor.trim() || !['NOT_SUBMITTED', 'RETURNED'].includes(acceptance?.status || '')} onClick={() => record('SUBMITTED')}>Ghi nhận nộp nghiệm thu</button>
        <button disabled={!actor.trim() || !reason.trim() || acceptance?.status !== 'SUBMITTED'} onClick={() => record('RETURNED')}>Ghi nhận trả lại</button>
        <button disabled={!actor.trim() || acceptance?.status !== 'SUBMITTED'} onClick={() => record('ACCEPTED')}>Ghi nhận chấp nhận</button>
      </div>
      <p className="muted">Đây là quyết định nghiệm thu do người dùng ghi nhận. So sánh số vòng trả lại trên các dataset baseline/pilot tương đương, cùng tiêu chí và quy mô.</p>
      {acceptance?.events.map(e => <p key={e.id}>Lần nộp {e.cycle} · {e.status} · {e.actor} · {e.reason || 'Không có ghi chú'}</p>)}
    </fieldset>
  </details>;
}
