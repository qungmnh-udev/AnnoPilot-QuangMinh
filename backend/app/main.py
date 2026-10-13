from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import initialize_database
from app.routers.api import router
from app.routers.evaluation import router as evaluation_router

initialize_database()
app = FastAPI(title='AnnoPilot', description='Model-assisted QC: missing objects, wrong classes, independent baseline/pilot evaluation and human review.', version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000','http://127.0.0.1:3000'],allow_methods=['*'],allow_headers=['*'])
app.include_router(router)
app.include_router(evaluation_router)

@app.get('/')
def root():
    return {
        'service': 'AnnoPilot Backend (Model-Assisted QC)',
        'version': '1.0.0',
        'status': 'healthy',
        'frontend_url': 'http://localhost:3000',
        'api_docs': '/docs',
        'api_prefix': '/api'
    }
