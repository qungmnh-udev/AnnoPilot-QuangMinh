#!/usr/bin/env python3
"""
CLI Tool for Option B Triad Inference & Ensemble
M1: YOLOv8n (1-Stage CNN)
M2: Faster R-CNN MobileNetV3 (2-Stage CNN)
M3: RT-DETR-l (Vision Transformer)

Usage:
    python backend/scripts/run_triad_inference.py --dataset-id 4 --limit 20
    python backend/scripts/run_triad_inference.py --dataset-id 4 --auto-ensemble
"""

import sys
import argparse
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal
from app.services.triad_detector import run_dataset_triad_inference, TRIAD_SPECS


def main():
    parser = argparse.ArgumentParser(description="Option B Triad Inference & WBF Ensemble")
    parser.add_argument("--dataset-id", type=int, required=True, help="Target Dataset ID")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames (for quick test)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.55, help="IoU clustering threshold for WBF (default: 0.55)")
    parser.add_argument("--no-ensemble", action="store_true", help="Do not run WBF ensemble automatically")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "fasterrcnn_mobilenet_v3", "rtdetr_l"],
                        help="List of models to run (default: yolov8n fasterrcnn_mobilenet_v3 rtdetr_l)")
    args = parser.parse_args()

    print("=" * 70)
    print("   ANNO-PILOT: OPTION B TRIAD MULTI-MODEL INFERENCE")
    print("   M1: YOLOv8n (1-Stage CNN)")
    print("   M2: Faster R-CNN MobileNetV3 (2-Stage CNN)")
    print("   M3: RT-DETR-l (Vision Transformer)")
    print("=" * 70)
    print(f"Target Dataset ID: {args.dataset_id}")
    print(f"Selected Models  : {', '.join(args.models)}")
    print(f"Conf Threshold   : {args.conf}")
    print(f"IoU Threshold    : {args.iou}")
    print(f"Frame Limit      : {args.limit or 'All'}")
    print(f"Auto-Ensemble    : {not args.no_ensemble}")
    print("-" * 70)

    db = SessionLocal()
    try:
        res = run_dataset_triad_inference(
            db=db,
            dataset_id=args.dataset_id,
            models=args.models,
            conf_thresh=args.conf,
            sample_limit=args.limit,
            auto_ensemble=not args.no_ensemble,
            iou_thresh=args.iou
        )

        print("\nINFERENCE COMPLETED:")
        print(f"  Frames Processed : {res['frames_processed']}")
        print(f"  Total Predictions: {res['total_predictions']}")
        for m, cnt in res["model_counts"].items():
            arch = TRIAD_SPECS.get(m, {}).get("family", "")
            print(f"    - {m:<24} [{arch:<18}]: {cnt:>5} boxes")

        ens = res.get("ensemble_result")
        if ens:
            print("\nWBF TRI-TIER CONSENSUS:")
            print(f"  Fused Boxes  : {ens.get('total_fused_predictions', 0)}")
            tb = ens.get("tier_breakdown", {})
            print(f"    * Tier 1 (High Consensus)    : {tb.get('tier_1_high_consensus', 0)} (Auto-accept)")
            print(f"    * Tier 2 (Class Disagree)    : {tb.get('tier_2_class_disagreement', 0)} (1-Click review)")
            print(f"    * Tier 3 (Model Divergence)  : {tb.get('tier_3_model_divergence', 0)} (QC queue)")
        print("\nAll predictions are now saved and ready in AnnoPilot QC!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
