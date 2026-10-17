"""Run real Multi-Model WBF vs single inference without opening the operational database."""
import argparse
import json
import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault('YOLO_CONFIG_DIR', str(BACKEND.parent / '.runtime' / 'ultralytics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=BACKEND.parent / 'demo_data' / 'images')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--golden', type=Path, help='Independent Golden JSON; metrics remain null if omitted')
    parser.add_argument('--annotations', type=Path, help='Annotation snapshot JSON with samples[].annotations[] for QC evaluation')
    parser.add_argument('--output', type=Path, default=BACKEND / 'benchmarks' / 'wbf-perturbations.json')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from app.schemas.experiments import WBFExperimentRequest
    from app.schemas.evaluation import BenchmarkRequest
    from app.services.wbf_experiments import run_experiment
    config = WBFExperimentRequest(sample_limit=args.limit, repeats=args.repeats, seed=args.seed)
    golden = None
    if args.golden:
        verified = BenchmarkRequest.model_validate(json.loads(args.golden.read_text(encoding='utf-8')))
        golden = {f.file_name: [a.model_dump() for a in f.annotations] for f in verified.samples}
    annotation_map = {}
    if args.annotations:
        annotation_map = {f['file_name']: f['annotations'] for f in json.loads(args.annotations.read_text(encoding='utf-8'))['samples']}
    paths = sorted(p for p in args.images.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if len({p.name for p in paths}) != len(paths):
        parser.error('Image basenames must be unique')
    if golden is not None:
        missing = set(golden) - {p.name for p in paths}
        if missing:
            parser.error(f'Golden images missing: {sorted(missing)}')
        paths = [p for p in paths if p.name in golden]
    if args.annotations:
        missing_annotations = {p.name for p in paths[:args.limit]} - set(annotation_map)
        if missing_annotations:
            parser.error(f'Annotation snapshot missing frames (record empty annotations explicitly): {sorted(missing_annotations)}')
    frames = [dict(file_name=p.name, image_path=str(p),
                   annotations_available=bool(args.annotations),
                   annotations=[dict(id=i, shape_type='BBOX_2D', **a) for i, a in enumerate(annotation_map.get(p.name, []))],
                   golden=golden[p.name] if golden is not None else None) for p in paths[:args.limit]]
    report = run_experiment(frames, config, args.output, progress=lambda state: print(json.dumps(state), flush=True))
    print(json.dumps(dict(status=report['status'], error=report['error'], summary=report['summary']), indent=2), flush=True)
    return 0 if report['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
