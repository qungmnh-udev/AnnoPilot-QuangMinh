from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Dataset, GoldenBenchmark, QCEvaluation
from app.schemas.evaluation import BenchmarkRequest, EvaluateRequest, AcceptanceRequest
from app.services.evaluation import create_benchmark, compare_strategies, record_acceptance, acceptance_summary

router = APIRouter(prefix='/api/datasets/{dataset_id}/qc')


def dataset_or_404(db, dataset_id):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, 'Dataset not found')
    return dataset


@router.post('/benchmarks')
def upload_benchmark(dataset_id: int, request: BenchmarkRequest, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    try:
        return create_benchmark(db, dataset, request)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get('/evaluations')
def evaluations(dataset_id: int, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    return dict(benchmarks=[dict(id=b.id, verified_by=b.verified_by, frame_count=len(b.samples), created_at=b.created_at.isoformat()) for b in dataset.benchmarks],
                runs=[dict(id=r.id, benchmark_id=r.benchmark_id, strategy=r.strategy, metrics=r.metrics,
                           created_at=r.created_at.isoformat()) for r in sorted(dataset.evaluations, key=lambda r: -r.id)])


@router.post('/evaluations/compare')
def compare(dataset_id: int, request: EvaluateRequest, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    try:
        return compare_strategies(db, dataset, request)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(422, str(exc))


@router.get('/evaluations/{evaluation_id}')
def evaluation_detail(dataset_id: int, evaluation_id: int, db: Session = Depends(get_db)):
    dataset_or_404(db, dataset_id)
    run = db.get(QCEvaluation, evaluation_id)
    if run is None or run.dataset_id != dataset_id:
        raise HTTPException(404, 'Evaluation not found')
    return dict(id=run.id, benchmark_id=run.benchmark_id, strategy=run.strategy,
                config=run.config, metrics=run.metrics, input_snapshot=run.input_snapshot)


@router.get('/acceptance')
def acceptance(dataset_id: int, db: Session = Depends(get_db)):
    return acceptance_summary(dataset_or_404(db, dataset_id))


@router.post('/acceptance')
def transition(dataset_id: int, request: AcceptanceRequest, db: Session = Depends(get_db)):
    dataset = dataset_or_404(db, dataset_id)
    try:
        return record_acceptance(db, dataset, request)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
