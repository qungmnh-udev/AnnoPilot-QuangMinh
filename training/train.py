"""CLI for the same training jobs used by Model QC. No separate legacy trainer."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))


def main():
    from app.database import SessionLocal, initialize_database
    from app.models import Dataset
    from app.schemas.training import TrainingRequest
    from app.services.training import MODELS, start_training
    parser = argparse.ArgumentParser(description='Train a detector and automatically register its checkpoint for QC.')
    parser.add_argument('--dataset-id', type=int, required=True)
    parser.add_argument('--model', choices=list(MODELS), default='yolo26n')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
    parser.add_argument('--labels-reviewed', action='store_true', required=True)
    args = parser.parse_args()
    initialize_database()
    with SessionLocal() as db:
        dataset = db.get(Dataset, args.dataset_id)
        if dataset is None:
            parser.error('Dataset does not exist')
        job = start_training(db, dataset, TrainingRequest(model_name=args.model, epochs=args.epochs,
                             device=args.device, labels_reviewed=args.labels_reviewed))
        print(f"Training job: {job['id']}. Open Model QC to follow progress.")


if __name__ == '__main__':
    main()
