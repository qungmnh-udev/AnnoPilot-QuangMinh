import { useEffect, useState } from 'react';
import { api, json } from '../api';
import type { YOLOVariant } from '../types';

type Metric = { precision: number | null; recall: number | null; f1: number | null };
type Summary = { frame_count: number; detections: number; detector_calls: number; inference_seconds: number;
  fusion_seconds: number; qc_seconds: number; qc_issues: number; detection_metrics: Metric | null;
  qc_metrics: Record<string, Metric> | null; agreement_with_single_once: Metric };
type Experiment = { id: string; status: string; processed_frames: number; total_frames: number;
  current_frame?: string; arm?: string; error?: string; summary: Record<string, Summary>;
  config?: { repeats: number; seed: number; baseline_model: string }; quality_measured?: boolean };
const armNames: Record<string, string> = {
  single_model_once: '1 model · 1 lượt gốc',
  single_model_repeated_clean: '1 model · lặp ảnh gốc',
  single_model_repeated_perturbed: '1 model · gốc + nhiễu',
  multi_model_once: 'Nhiều model · 1 lượt gốc',
  multi_model_repeated_clean: 'Nhiều model · lặp ảnh gốc',
  multi_model_repeated_perturbed: 'Nhiều model · gốc + nhiễu',
};
const pct = (v: number | null | undefined) => v == null ? '—' : `${(v * 100).toFixed(1)}%`;

export function WBFExperimentPanel({ datasetId, yoloModel }: { datasetId: number; yoloModel: YOLOVariant }) {
  const [runs, setRuns] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState('');
  const [benchmarks, setBenchmarks] = useState<{ id: number; frame_count: number; verified_by: string }[]>([]);
  const [benchmark, setBenchmark] = useState('');
  const [limit, setLimit] = useState(5);
  const [repeats, setRepeats] = useState(4);
  const [seed, setSeed] = useState(42);
  const [sigma, setSigma] = useState(8);
  const [brightness, setBrightness] = useState(.15);
  const [blur, setBlur] = useState(.7);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const prefix = `/datasets/${datasetId}/ensemble/experiments`;
  const running = runs.some(run => run.status === 'RUNNING');
  const run = runs.find(item => item.id === selected) || runs[0];

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const [items, evaluation] = await Promise.all([
          api<Experiment[]>(prefix), api<{ benchmarks: typeof benchmarks }>(`/datasets/${datasetId}/qc/evaluations`),
        ]);
        if (alive) { setRuns(items); setBenchmarks(evaluation.benchmarks); }
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : 'Không tải được thử nghiệm'); }
      if (alive) timer = setTimeout(refresh, 2500);
    }
    void refresh();
    return () => { alive = false; clearTimeout(timer); };
  }, [datasetId, prefix]);

  async function start(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const result = await api<Experiment>(prefix, json('POST', {
        baseline_model: yoloModel, models: [yoloModel, 'fasterrcnn_mobilenet_v3', 'rtdetr_l'],
        sample_limit: limit, repeats, seed, gaussian_sigma: sigma, brightness_delta: brightness, blur_radius: blur,
        benchmark_id: benchmark ? Number(benchmark) : null,
      }));
      setSelected(result.id); setRuns(current => [{ ...result, processed_frames: 0, summary: {} }, ...current]);
    } catch (e) { setError(e instanceof Error ? e.message : 'Không khởi động được thử nghiệm'); }
    finally { setBusy(false); }
  }

  return <section className="wbf-experiment-panel" aria-labelledby="wbf-experiment-title">
    <h3 id="wbf-experiment-title">Thử nghiệm WBF thực tế: một lượt và nhiều lượt có nhiễu</h3>
    <p>So sánh 6 cấu hình: một model hoặc nhiều model, mỗi cấu hình chạy một lượt gốc, lặp ảnh gốc hoặc lặp có nhiễu. Mặc định 4 lượt: gốc, Gaussian, tăng độ sáng và làm mờ nhẹ.</p>
    <p className="muted">Các lượt của cùng model được gộp trước khi WBF giữa các model. Kết quả thử nghiệm lưu riêng, không sửa nhãn hoặc prediction đang review. Đóng cửa sổ vẫn tiếp tục chạy; khởi động lại backend sẽ ngắt thử nghiệm.</p>
    {error && <p role="alert" className="text-warning">{error}</p>}
    <form onSubmit={start}>
      <fieldset disabled={busy || running}>
        <div className="wbf-experiment-controls">
          <label>Số ảnh (tối đa 30)<input type="number" min="1" max="30" required value={limit} onChange={e => setLimit(Number(e.target.value))} /></label>
          <label>Số lượt gồm ảnh gốc<input type="number" min="2" max="8" required value={repeats} onChange={e => setRepeats(Number(e.target.value))} /></label>
          <label>Seed<input type="number" min="0" max="4294967295" required value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
          <label>Gaussian sigma (pixel 0–255)<input type="number" min="0" max="30" step="0.5" required value={sigma} onChange={e => setSigma(Number(e.target.value))} /></label>
          <label>Độ sáng tăng (0,15 = 15%)<input type="number" min="0" max="0.5" step="0.01" required value={brightness} onChange={e => setBrightness(Number(e.target.value))} /></label>
          <label>Bán kính làm mờ<input type="number" min="0" max="3" step="0.1" required value={blur} onChange={e => setBlur(Number(e.target.value))} /></label>
        </div>
        <label>Golden để đo chất lượng<select value={benchmark} onChange={e => setBenchmark(e.target.value)}>
          <option value="">Chưa có Golden — chỉ đo tốc độ và mức nhất quán</option>
          {benchmarks.map(b => <option key={b.id} value={b.id}>#{b.id} · {b.frame_count} ảnh · {b.verified_by}</option>)}
        </select></label>
        <p className="muted">Cần 4 lượt để có đủ ba loại nhiễu; lượt 5 trở đi lặp chu kỳ. Confidence cố định 0,35; fusion IoU 0,55; các model có weight bằng nhau.</p>
        <button type="submit" className="primary">{busy || running ? 'Đang chạy model thật…' : 'Chạy so sánh inference thật'}</button>
      </fieldset>
    </form>
    {!!runs.length && <label>Lịch sử thử nghiệm<select value={run?.id || ''} onChange={e => setSelected(e.target.value)}>
      {runs.map(item => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.status} · {item.config?.baseline_model || yoloModel}</option>)}
    </select></label>}
    {run && <>
      <p role="status"><strong>{run.status}</strong> · {run.processed_frames}/{run.total_frames} ảnh
        {run.config && ` · ${run.config.repeats} lượt · seed ${run.config.seed}`}
        {run.current_frame && ` · ${run.current_frame} · ${armNames[run.arm || ''] || ''}`}</p>
      {run.error && <p role="alert" className="text-warning">{run.error}</p>}
      {!run.quality_measured && <p className="text-warning">Chưa dùng Golden độc lập: chưa thể kết luận WBF hoặc thêm nhiễu chính xác hơn. F1 nhất quán với baseline không phải F1 chất lượng.</p>}
      <div className="table-wrap"><table><thead><tr><th>Cấu hình</th><th>Ảnh / calls</th><th>Boxes</th><th>Inference / fusion / QC (s)</th><th>Detection P / R / F1</th><th>QC recall thiếu / sai class</th><th>F1 nhất quán với baseline</th></tr></thead>
        <tbody>{Object.entries(run.summary || {}).map(([arm, s]) => <tr key={arm}>
          <td>{armNames[arm] || arm}</td><td>{s.frame_count} / {s.detector_calls}</td><td>{s.detections}</td>
          <td>{s.inference_seconds.toFixed(2)} / {s.fusion_seconds.toFixed(3)} / {s.qc_seconds.toFixed(3)}</td>
          <td>{pct(s.detection_metrics?.precision)} / {pct(s.detection_metrics?.recall)} / {pct(s.detection_metrics?.f1)}</td>
          <td>{pct(s.qc_metrics?.MISSING_OBJECT?.recall)} / {pct(s.qc_metrics?.WRONG_CLASS?.recall)}</td>
          <td>{pct(s.agreement_with_single_once.f1)}</td>
        </tr>)}</tbody></table></div>
      <a className="button secondary" href={`/api${prefix}/${run.id}/report`}>Tải prediction từng lượt và báo cáo JSON</a>
    </>}
  </section>;
}
