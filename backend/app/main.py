from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import initialize_database
from app.routers.api import router

initialize_database()
app = FastAPI(title='AnnoPilot',description='Intelligent Annotation Review Assistant. Difficulty estimates effort, not correctness.',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000','http://127.0.0.1:3000'],allow_methods=['*'],allow_headers=['*'])
app.include_router(router)

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
