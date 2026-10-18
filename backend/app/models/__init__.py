from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.database import Base

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    format = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    storage_key = Column(String, nullable=True)
    cvat_task_id = Column(Integer, nullable=True)
    cvat_job_id = Column(Integer, nullable=True)
    cvat_base_url = Column(String, nullable=True, default='http://localhost:8080')
    cvat_token = Column(String, nullable=True, default='')
    warnings = Column(JSON, default=list)
    revision = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    samples = relationship('Sample', back_populates='dataset', cascade='all, delete-orphan')
    benchmarks = relationship('GoldenBenchmark', cascade='all, delete-orphan')
    evaluations = relationship('QCEvaluation', cascade='all, delete-orphan')
    acceptance_events = relationship('AcceptanceEvent', cascade='all, delete-orphan')

class Sample(Base):
    __tablename__ = 'samples'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    file_name = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    media_path = Column(String, nullable=True)
    frame_number = Column(Integer, nullable=True)
    inference_state = Column(JSON, default=dict)
    task_type = Column(String, nullable=False)
    dataset = relationship('Dataset', back_populates='samples')
    annotations = relationship('Annotation', cascade='all, delete-orphan')
    predictions = relationship('Prediction', cascade='all, delete-orphan')
    qc_issues = relationship('QCIssue', cascade='all, delete-orphan')
    review_events = relationship('IssueReviewEvent', cascade='all, delete-orphan')

class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    label = Column(String, nullable=False)
    shape_type = Column(String, nullable=False)
    geometry = Column(JSON, nullable=False)
    attributes = Column(JSON, default=dict)
    occluded = Column(Boolean, nullable=True)
    source_metadata = Column(JSON, default=dict)

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(Integer, primary_key=True, default=1)
    config = Column(JSON, nullable=False)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    label = Column(String, nullable=False)
    geometry = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    source_model = Column(String, default='pretrained')
    extra_data = Column(JSON, default=dict)

class QCIssue(Base):
    __tablename__ = 'qc_issues'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    issue_type = Column(String, nullable=False)  # 'MISSING_OBJECT' | 'WRONG_CLASS'
    location = Column(JSON, nullable=False)
    human_label = Column(String, nullable=True)
    suggested_label = Column(String, nullable=False)
    annotation_id = Column(Integer, ForeignKey('annotations.id'), nullable=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=True)
    qc_score = Column(Float, nullable=False)
    evidence = Column(JSON, default=dict)
    frame_number = Column(Integer, nullable=True)
    cvat_url = Column(String, nullable=True)
    status = Column(String, default='PENDING', nullable=False)  # 'PENDING' | 'RESOLVED' / 'ACCEPTED' | 'FALSE_POSITIVE' / 'REJECTED'
    reviewer_note = Column(String, default='', nullable=False)


class GoldenBenchmark(Base):
    __tablename__ = 'golden_benchmarks'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    verified_by = Column(String, nullable=False)
    notes = Column(String, default='')
    samples = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QCEvaluation(Base):
    __tablename__ = 'qc_evaluations'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    benchmark_id = Column(Integer, ForeignKey('golden_benchmarks.id'), nullable=False)
    strategy = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    input_snapshot = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AcceptanceEvent(Base):
    __tablename__ = 'acceptance_events'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    cycle = Column(Integer, nullable=False)
    strategy = Column(String, nullable=False)
    status = Column(String, nullable=False)
    actor = Column(String, nullable=False)
    reason = Column(String, default='')
    annotation_snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class IssueReviewEvent(Base):
    __tablename__ = 'issue_review_events'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    issue_id = Column(Integer, nullable=False)  # history survives issue reconciliation
    previous_status = Column(String, nullable=False)
    status = Column(String, nullable=False)
    reviewer_note = Column(String, default='')
    evidence_snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TrainingJob(Base):
    __tablename__ = 'training_jobs'
    id = Column(String, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    model_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default='QUEUED')
    # A unique occupied slot serializes training across API processes/restarts.
    active_slot = Column(String, nullable=True, unique=True)
    config = Column(JSON, nullable=False)
    progress = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelCheckpoint(Base):
    __tablename__ = 'model_checkpoints'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    job_id = Column(String, ForeignKey('training_jobs.id'), nullable=False, unique=True)
    model_name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    metrics = Column(JSON, default=dict)
    training_images = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
