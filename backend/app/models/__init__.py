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
    warnings = Column(JSON, default=list)
    revision = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    samples = relationship('Sample', cascade='all, delete-orphan')
    runs = relationship('SamplingRun', cascade='all, delete-orphan')
    reviewers = relationship('Reviewer', cascade='all, delete-orphan')

class Sample(Base):
    __tablename__ = 'samples'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    file_name = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    media_path = Column(String, nullable=True)
    task_type = Column(String, nullable=False)
    annotations = relationship('Annotation', cascade='all, delete-orphan')
    analysis = relationship('AnalysisResult', uselist=False, cascade='all, delete-orphan')
    predictions = relationship('Prediction', cascade='all, delete-orphan')
    qc_issues = relationship('QCIssue', cascade='all, delete-orphan')

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

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), unique=True, nullable=False)
    details = Column(JSON, nullable=False)
    annotation_difficulty = Column(Float, nullable=True)
    visual_difficulty = Column(Float, nullable=True)
    overall_difficulty = Column(Float, nullable=True)
    level = Column(String, nullable=False)

class SamplingRun(Base):
    __tablename__ = 'sampling_runs'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    revision = Column(Integer, nullable=False)
    config = Column(JSON, nullable=False)
    selections = relationship('SampleSelection', cascade='all, delete-orphan')

class SampleSelection(Base):
    __tablename__ = 'sample_selections'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('sampling_runs.id'), nullable=False)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    reason = Column(String, nullable=False)
    assignment = relationship('ReviewAssignment', uselist=False, cascade='all, delete-orphan')

class Reviewer(Base):
    __tablename__ = 'reviewers'
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    name = Column(String, nullable=False)
    assignments = relationship('ReviewAssignment', cascade='all, delete-orphan')

class ReviewAssignment(Base):
    __tablename__ = 'review_assignments'
    id = Column(Integer, primary_key=True)
    selection_id = Column(Integer, ForeignKey('sample_selections.id'), unique=True, nullable=False)
    reviewer_id = Column(Integer, ForeignKey('reviewers.id'), nullable=False)
    status = Column(String, default='PENDING', nullable=False)
    note = Column(String, default='', nullable=False)
    revision = Column(Integer, nullable=False)

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
    status = Column(String, default='PENDING', nullable=False)
    reviewer_note = Column(String, default='', nullable=False)

