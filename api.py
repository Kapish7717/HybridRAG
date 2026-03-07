from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
import tempfile
from datetime import datetime
from ingestion import ingest_pdf
from rag import generate_answer, refresh_bm25
from typing import Dict, List, Tuple, Any,Optional
import time
import json
import asyncio

app_start_time = datetime.now()
total_queries = 0
total_uploads = 0
total_errors = 0


app = FastAPI()

UPLOAD_DIR = "uploads"
CHROMA_PATH = "./chroma"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.on_event('startup')
# async def startup_event():
#     print("🚀 Starting up RAG system...")

#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     os.makedirs(CHROMA_PATH, exist_ok=True)

#     print("📦 Loading vector database...")
#     get_vectordb()

#     print("✅ Startup complete! System is warm and ready.")

chat_sessions={}

@app.get('/metrics')
def get_metrics():
    """System performance metrics"""
    uptime = datetime.now() - app_start_time
    
    return {
        "system": {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_human": str(uptime).split('.')[0],
            "status": "healthy"
        },
        "usage": {
            "total_queries": total_queries,
            "total_uploads": total_uploads,
            "total_errors": total_errors,
            "active_sessions": len(chat_sessions),
            "error_rate": f"{(total_errors/max(total_queries,1)*100):.1f}%"
        }
    }

@app.get('/')
def health_check():
    return{
        "status":"healthy",
        "message":"API is running"
    }


class QueryRequest(BaseModel):
    query:str
    doc_id:Optional[str]=None
    session_id:Optional[str]=None

class QueryResponse(BaseModel):
    query:str
    answer:str
    session_id: str
    sources:List[Dict[str,Any]]=[]

class ConversationTurn(BaseModel):
    turn: int
    user_message: str
    assistant_response: str


class SessionDebugResponse(BaseModel):
    session_id :str
    exists:bool
    message_count: int
    conversation:List[ConversationTurn]

class RegisterRequest(BaseModel):
    username:str
    password:str


@app.post('/ingest')
def ingest_document(file:UploadFile=File(...),
    # username: str = Depends(verify_token) 
    ):
    # print(f"👤 User {username} is uploading {file.filename}")
    # if not file.filename.endswith('.pdf'):
    #     raise HTTPException(
    #         status_code=400,
    #         detail='ONLY PDFs ARE ACCEPTED!'
    #     )
    username="default"
    try:
        # Create a temporary file instead of saving to uploads folder
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        result = ingest_pdf(temp_file_path)
        
        # Refresh the BM25 index now that new documents are in the DB
        refresh_bm25()
        
        try:
            os.remove(temp_file_path)
            print(f"🗑️ Cleaned up temp file: {temp_file_path}")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")
        
        chat_sessions.clear()
        print("Cleared all chat sessions!!")
        return {
            "status": "success",
            "doc_id": result["doc_id"],
            "filename": result["filename"],
            "chunks": result["chunks"],
            "message": f"Successfully processed {result['total_pages']} pages into {result['chunks']} chunks"
        }
    except ValueError as e:
        # Handle PDF processing errors
        raise HTTPException(
            status_code=400,
            detail=f"PDF processing failed: {str(e)}"
        )
    
    except Exception as e:
        # Handle unexpected errors
        print(f"❌ Ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )
        
        

@app.post('/query')
def query_documents(request:QueryRequest):
    username="default"
    try:
        session_id = request.session_id or str(uuid.uuid4())
        session_key = f"{username}:{session_id}"
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]

        answer, sources, raw_contexts = generate_answer(request.query, chat_history)

        # chat_sessions will store: (user_query, assistant_answer)
        chat_sessions[session_id].append((request.query, answer))
        if len(chat_sessions[session_id]) > 10:
            chat_sessions[session_id] = chat_sessions[session_id][-10:]
            
        return {
            "query": request.query,
            "answer": answer,
            "session_id":session_id
        }
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

# Removed get_session_metrics endpoint
