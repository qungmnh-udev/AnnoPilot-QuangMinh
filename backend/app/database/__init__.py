import os
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parents[3]
UPLOADS = Path(os.getenv('UPLOAD_DIR', str(ROOT / 'uploads')))
EXPORTS = Path(os.getenv('EXPORT_DIR', str(ROOT / 'exports')))
UPLOADS.mkdir(parents=True, exist_ok=True)
EXPORTS.mkdir(parents=True, exist_ok=True)
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///' + str(ROOT / 'annopilot.db')), connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    with SessionLocal() as session:
        yield session

def initialize_database():
    import app.models  # noqa: F401
    Base.metadata.create_all(engine)
    # Additive migration for schema updates; preserves all existing user data.
    insp = inspect(engine)
    table_names = insp.get_table_names()
    with engine.begin() as connection:
        if 'datasets' in table_names:
            cols = {col['name'] for col in insp.get_columns('datasets')}
            if 'storage_key' not in cols:
                connection.execute(text('ALTER TABLE datasets ADD COLUMN storage_key VARCHAR'))
            if 'cvat_task_id' not in cols:
                connection.execute(text('ALTER TABLE datasets ADD COLUMN cvat_task_id INTEGER'))
            if 'cvat_job_id' not in cols:
                connection.execute(text('ALTER TABLE datasets ADD COLUMN cvat_job_id INTEGER'))
            if 'cvat_base_url' not in cols:
                connection.execute(text("ALTER TABLE datasets ADD COLUMN cvat_base_url VARCHAR DEFAULT 'http://localhost:8080'"))

        if 'samples' in table_names:
            cols = {col['name'] for col in insp.get_columns('samples')}
            if 'frame_number' not in cols:
                connection.execute(text('ALTER TABLE samples ADD COLUMN frame_number INTEGER'))

        if 'qc_issues' in table_names:
            cols = {col['name'] for col in insp.get_columns('qc_issues')}
            if 'frame_number' not in cols:
                connection.execute(text('ALTER TABLE qc_issues ADD COLUMN frame_number INTEGER'))
            if 'cvat_url' not in cols:
                connection.execute(text('ALTER TABLE qc_issues ADD COLUMN cvat_url VARCHAR'))
