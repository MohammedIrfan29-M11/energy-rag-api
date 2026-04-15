import os
import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.pdf_service import process_pdf
from app.core.config import UPLOAD_DIR,ALLOWED_EXTENSIONS,MAX_FILE_SIZE_MB
from app.services.rag_service import SIMILARITY_THRESHOLD, query_rag
from app.services.embedding_service import embed_chunks,get_collection_count,collection

logger=logging.getLogger('app.routes')

router=APIRouter()

class QueryRequest(BaseModel):
    question:str
    history:list[dict]=[]
    class config:
        json_schema_extra={
            "example":{
                "question":"What are the renewable energy targets of 2030",
                "history":[]
            }
        }

class CitationResponse(BaseModel):
    source: str
    similarity: float
    preview: str

class QueryResponse(BaseModel):
    answer:str
    citations:list[dict]
    chunks_used:int
    question:str
    history:list[dict]

class DocumentInfo(BaseModel):
    filename: str
    size_kb: float
    chunks_stored: int

@router.get("/health")
def health_check():
    """Health Check Endpoint
       Returns system status including how many chunks indexed
    """
    try:
        chunk_count=get_collection_count()
        return{
            "status":"healthy",
            "chunks_indexed":chunk_count,
            "ready_for_queries":chunk_count>0
        }
    except Exception as e:
        return{
            "status":"degraded",
            "error":str(e)
        }
    
@router.post("/documents/upload")
async def upload_documents(file: UploadFile = File(...)):
    """
    Upload a PDF document and index it for RAG queries.

    Pipeline:
    1. Validate file type and size
    2. Save to documents/ folder
    3. Extract text with pypdf
    4. Chunk into ~300 token pieces
    5. Embed chunks and store in ChromaDB

    Returns chunk count and processing stats.
    """
    logger.info(f"Upload Request | filename: {file.filename}")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File Type {file_ext} not supported."
        )
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"file too large {file_size_mb:.1f}. Max allowed {MAX_FILE_SIZE_MB}"
        )
    file_path = os.path.join(UPLOAD_DIR,file.filename)
    with open(file_path,"wb") as f:
        f.write(content)

    logger.info(f"file saved | path: {file_path} | size: {file_size_mb:.2f}MB")

    try:
        chunks = process_pdf(file_path)
    except ValueError as e:
        os.remove(file_path)
        raise HTTPException(status_code=402,detail=str(e))
    
    try:
        embed_chunks(chunks)
    except Exception as e:
        logger.error(f"Embedding failed | {str(e)}")
        raise HTTPException(status_code=422,detail=f"Embedding failed: {str(e)}")
    
    logger.info(
        f"Upload Complete |"
        f"document length {len(chunks)} |"
        f"file name: {file.filename} |"
        f"file size: {file_size_mb:.2f}MB"
    )

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2),
        "chunks_created": len(chunks),
        "total_chunks_indexed": get_collection_count()
    }


@router.get("/documents")
def list_documents():
    """List all documents in the documents/folder.
       Shows file name and size
    """
    try:
        files=[]
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith('.pdf'):
                file_path= os.path.join(UPLOAD_DIR,filename)
                size_kb = os.path.getsize(file_path) / 1024
                files.append({
                    "filename":filename,
                    "size_kb":round(size_kb,1)
                })
        return {
            "documents":files,
            "total_documents":len(files),
            "total_chunks_indexed":get_collection_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@router.post("/rag/query",response_model=QueryResponse)
def rag_query(request: QueryRequest):
    """
    Query the indexed documents using RAG

    Pipeline:
    1. Validate Question
    2. Search ChromaDB for relevant chunks
    3. filter by similarity threshold(0.65)
    4. build grounded prompt
    5. send to claude with context
    6. returns answers with citations

    if no relevant chunks found - returns message.
    if ChromaDB is empty - returns helpful error.
    """

    logger.info(f"Query Request | question: '{request.question[:60]}'")

    question=request.question.strip()
    if not question:
        raise HTTPException(status_code=400,detail="Question can not be empty")
    if len(question) > 2000:
        raise HTTPException(status_code=500,details="Question too long, please enter a question not more than 2000 words")
        
    chunk_count = get_collection_count()
    if chunk_count==0:
        raise HTTPException(status_code=400,detail="Chunks not indexed, please upload documents")
        

    try:
        result=query_rag(question,request.history)
    except Exception as e:
        logger.error(f"query failed | {str(e)}")
        raise HTTPException(status_code=500,details=str(e))

    return QueryResponse(
        answer=result['answer'],
        citations=result['citations'],
        chunks_used=len(result['citations']),
        question=question,
        history=result['history']
    )
