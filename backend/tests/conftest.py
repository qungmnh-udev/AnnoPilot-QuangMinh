import os
from pathlib import Path
import tempfile

# Test-generated datasets never enter production storage.
TEST_ROOT = Path(tempfile.mkdtemp(prefix='annopilot-tests-',dir=Path(__file__).parent))
os.environ['DATABASE_URL'] = 'sqlite:///' + str(TEST_ROOT/'test.db')
os.environ['UPLOAD_DIR'] = str(TEST_ROOT/'uploads')
os.environ['EXPORT_DIR'] = str(TEST_ROOT/'exports')

import pytest
from fastapi.testclient import TestClient
from app.database import Base,engine
from app.main import app

@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app) as client:
        yield client

@pytest.fixture
def db_session():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from app.database import SessionLocal
    with SessionLocal() as session:
        yield session

def pytest_sessionfinish(session,exitstatus):
    import shutil
    engine.dispose()
    shutil.rmtree(TEST_ROOT,ignore_errors=True)
