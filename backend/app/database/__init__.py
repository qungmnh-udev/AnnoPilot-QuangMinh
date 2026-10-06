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
    # Additive migration for the initial MVP schema; preserves any existing data.
    table_names = inspect(engine).get_table_names()
    if 'datasets' in table_names:
        columns = {column['name'] for column in inspect(engine).get_columns('datasets')}
        if 'storage_key' not in columns:
            with engine.begin() as connection:
                connection.execute(text('ALTER TABLE datasets ADD COLUMN storage_key VARCHAR'))
