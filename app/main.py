import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.logging_config import setup_logging
from app.core.config import UPLOAD_DIR,CHROMA_DB_PATH

setup_logging()
logger=logging.getLogger('app.main')

os.makedirs(UPLOAD_DIR,exist_ok=True)
os.makedirs(CHROMA_DB_PATH,exist_ok=True)
os.makedirs('logs',exist_ok=True)

app= FastAPI(
    title="EnergyRAG — Energy Policy Document Intelligence",
    description="Upload energy policy PDFs and query them using Claude AI",
    version="1.0.0" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "name": "EnergyRAG API",
        "description": "Upload energy policy documents and query them using RAG with Claude AI",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /documents/upload",
            "list": "GET /documents",
            "query": "POST /rag/query",
            "health": "GET /health"
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("="*50)
    logger.info("Starting EnergyRAG API")
    logger.info("="*50)

