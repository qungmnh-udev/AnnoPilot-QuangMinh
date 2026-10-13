"""Benchmark actual YOLO26 inference in isolated processes, without opening the app DB."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))


def worker(args):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'backend' / '.runtime' / 'ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'backend' / '.runtime' / 'matplotlib'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    import psutil
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample_memory():
        while not stop.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    import_start = perf_counter()
    import torch
    from importlib.metadata import version
    from PIL import Image
    from app.services.yolo_detector import get_yolo_weights_path, load_yolo, inference_device
    device = inference_device(args.device)
    torch.set_num_threads(args.threads)
    import_seconds = perf_counter() - import_start
    model_name = args.worker
    images = sorted(p for p in Path(args.images_dir).glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:args.limit]
    if not images:
        raise ValueError('No images to benchmark')
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    load_started = perf_counter()
    model = load_yolo(model_name)
    load_seconds = perf_counter() - load_started
    def infer(image):
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        started = perf_counter()
        result = model.predict(str(image), device=device, imgsz=args.imgsz, conf=args.conf,
                               verbose=False, save=False)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        peak[0] = max(peak[0], process.memory_info().rss)
        return perf_counter() - started, len(result[0].boxes)
    first_seconds, _ = infer(images[0])
    for _ in range(args.warmup):
        infer(images[0])
    timings, counts = [], []
    for _ in range(args.repeat):
        for image in images:
            elapsed, count = infer(image)
            timings.append(elapsed)
            counts.append(count)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(.95 * len(ordered)))]
    with Image.open(images[0]) as image:
        dimensions = list(image.size)
    result = dict(model=model_name, status='success', checkpoint=str(get_yolo_weights_path(model_name)),
                  checkpoint_mb=round(get_yolo_weights_path(model_name).stat().st_size / 2**20, 2),
                  device=device, torch=torch.__version__, ultralytics=version('ultralytics'),
                  threads=torch.get_num_threads(), imgsz=args.imgsz, confidence=args.conf,
                  image_count=len(images), repetitions=args.repeat, measured_inferences=len(timings),
                  first_image_dimensions=dimensions, import_seconds=round(import_seconds, 3),
                  model_load_seconds=round(load_seconds, 3), first_inference_ms=round(first_seconds * 1000, 2),
                  warmup_inferences=args.warmup + 1, mean_ms=round(statistics.mean(timings) * 1000, 2),
                  median_ms=round(statistics.median(timings) * 1000, 2), p95_ms=round(p95 * 1000, 2),
                  throughput_fps=round(1 / statistics.mean(timings), 2),
                  peak_process_rss_mb=round(peak[0] / 2**20, 1),
                  peak_gpu_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.startswith('cuda') else None,
                  detections_per_image=counts, images=[p.name for p in images])
    stop.set(); sampler.join()
    Path(args.result).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['yolo26n', 'yolo26s'], default=['yolo26n', 'yolo26s'])
    parser.add_argument('--images-dir', default=str(ROOT / 'demo_data' / 'images'))
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.35)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', default=str(ROOT / 'backend' / 'benchmarks' / 'yolo26-hardware.json'))
    parser.add_argument('--worker', choices=['yolo26n', 'yolo26s'])
    parser.add_argument('--result')
    args = parser.parse_args()
    if min(args.limit, args.repeat, args.threads, args.imgsz) <= 0 or args.warmup < 0:
        parser.error('Counts/sizes must be positive; warmup must be nonnegative')
    if args.worker:
        worker(args)
        return
    import psutil
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  logical_cpus=os.cpu_count(), ram_total_gib=round(psutil.virtual_memory().total / 2**30, 2),
                  ram_available_before_gib=round(psutil.virtual_memory().available / 2**30, 2), results=[],
                  scope='Latency and resource feasibility only; no ground-truth detection/QC quality measurement')
    for name in args.models:
        result_path = output.with_name(f'{output.stem}-{name}.json')
        print(f'Benchmarking {name} on {args.device}...', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', name, '--result', str(result_path),
                   '--images-dir', args.images_dir, '--device', args.device, '--threads', str(args.threads),
                   '--imgsz', str(args.imgsz), '--conf', str(args.conf), '--limit', str(args.limit),
                   '--repeat', str(args.repeat), '--warmup', str(args.warmup)]
        completed = subprocess.run(command, text=True, capture_output=True, encoding='utf-8', errors='replace',
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if completed.returncode:
            report['results'].append(dict(model=name, status='error', error=completed.stderr[-4000:], stdout=completed.stdout[-2000:]))
            print(completed.stderr[-2000:], flush=True)
        else:
            result = json.loads(result_path.read_text(encoding='utf-8'))
            report['results'].append(result)
            print(f"{name}: mean={result['mean_ms']} ms, p95={result['p95_ms']} ms, peak RSS={result['peak_process_rss_mb']} MB", flush=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {output}', flush=True)
    if any(r['status'] != 'success' for r in report['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
