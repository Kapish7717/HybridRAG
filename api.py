from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
import tempfile
from datetime import datetime
from ingestion import ingest_pdf
from rag import generate_answer, refresh_bm25
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import asyncio
import sqlite3
from evaluation import evaluate_rag,generate_reference_answer
from text_to_sql import rank_tables,make_sql_prompt,extract_schema,generate_sql_query,sql_response,rag_response

app_start_time = datetime.now()
total_queries = 0
total_uploads = 0
total_errors = 0


app = FastAPI()

UPLOAD_DIR = "uploads"
CHROMA_PATH = "./chroma"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

chat_sessions = {}
current_db_path: Optional[str] = None  # Tracks the currently active database file path

# Global status tracking for background PDF ingestion
ingestion_state = {
    "status": "idle",
    "filename": None,
    "error": None,
    "doc_id": None,
    "chunks": 0,
    "total_pages": 0
}

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

@app.head('/')
@app.get('/')
def health_check():
    return{
        "status":"healthy",
        "message":"API is running"
    }


@app.get('/ready')
def readiness_check():
    """Readiness endpoint: ensures heavy models are loaded before declaring ready."""
    try:
        # Import here to avoid circular imports at module import time
        from rag import get_reranker
        from hallucination_check import load_hallucination_model

        # Trigger model initialization (should be fast if models were pre-downloaded at build time)
        get_reranker()
        load_hallucination_model()

        return JSONResponse(status_code=200, content={"ready": True})
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "error": str(e)})


class QueryRequest(BaseModel):
    query:str
    doc_id:Optional[str]=None
    session_id:Optional[str]=None
    ground_truth:Optional[str]=None
    answer:Optional[str]=None
    contexts:Optional[List[str]]=None

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

class SQLQueryRequest(BaseModel):
    query:str
    filename:str
    session_id:Optional[str]=None

class SQLQueryResponse(BaseModel):
    query: str
    generated_sql: str
    results: List[Dict[str, Any]]
    answer: str
    session_id: str
    error: Optional[str] = None


def run_ingestion_background(temp_file_path: str, filename: str):
    global ingestion_state
    try:
        result = ingest_pdf(temp_file_path)
        
        # Refresh the BM25 index now that new documents are in the DB
        refresh_bm25()
        
        ingestion_state.update({
            "status": "success",
            "doc_id": result["doc_id"],
            "chunks": result["chunks"],
            "total_pages": result["total_pages"]
        })
        print(f"[SUCCESS] Ingestion successful for {filename}")
    except Exception as e:
        print(f"[ERROR] Ingestion failed in background: {str(e)}")
        ingestion_state.update({
            "status": "failed",
            "error": str(e)
        })
    finally:
        try:
            os.remove(temp_file_path)
            print(f"[CLEANUP] Cleaned up temp file: {temp_file_path}")
        except Exception as e:
            print(f"[WARNING] Could not delete temp file: {e}")
        chat_sessions.clear()
        print("Cleared all chat sessions!!")


@app.post('/ingest')
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    global ingestion_state
    
    # Check if a background ingestion is already running
    if ingestion_state["status"] == "processing":
        raise HTTPException(
            status_code=400,
            detail="Another PDF ingestion is already in progress. Please wait."
        )
        
    try:
        # Create a temporary file instead of saving to uploads folder
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create temp file: {str(e)}"
        )

    # Initialize status tracking
    ingestion_state = {
        "status": "processing",
        "filename": file.filename,
        "error": None,
        "doc_id": None,
        "chunks": 0,
        "total_pages": 0
    }
    
    # Schedule the background task
    background_tasks.add_task(run_ingestion_background, temp_file_path, file.filename)
    
    return {
        "status": "processing",
        "message": "Ingestion started in the background"
    }


@app.get('/ingest/status')
def get_ingest_status():
    global ingestion_state
    return ingestion_state

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

@app.post('/evaluate')
def evaluate_query(request: QueryRequest):
    """
    Evaluates a RAG query using Ragas metrics: faithfulness, answer_relevancy, 
    context_precision, and context_recall.
    """
    try:
        # 1. Use manual input if provided, otherwise generate
        if request.answer and request.contexts:
            print("📝 Using manual answer and context for evaluation...")
            answer = request.answer
            raw_contexts = request.contexts
            sources = [] # No source metadata for manual contexts
        else:
            print("🔍 Generating RAG answer for evaluation...")
            answer, sources, raw_contexts = generate_answer(request.query)
        
        gt = request.ground_truth
        if not gt:
            gt = generate_reference_answer(request.query,raw_contexts)
        # 2. Run the evaluation
        # Ragas requires contexts to be a list of strings
        eval_metrics = evaluate_rag(
            query=request.query,
            answer=answer,
            contexts=raw_contexts,
            ground_truth=gt
        )

        return {
            "query": request.query,
            "answer": answer,
            "metrics": eval_metrics,
            "sources": sources,
            "ground_truth":gt
        }
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

# Removed get_session_metrics endpoint

@app.post('/upload-db')
async def upload_db(file:UploadFile=File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Database uploaded successfully!",
        "filename": file.filename
    }    

    


@app.post('/sql_query')
async def sql_query(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    global current_db_path
    
    try:
        if file:
            # If a new file is uploaded, clean up the old one
            if current_db_path and os.path.exists(current_db_path):
                try:
                    os.remove(current_db_path)
                except Exception:
                    pass
                    
            # Save the new file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                shutil.copyfileobj(file.file, tmp)
                current_db_path = tmp.name
        else:
            # If no file is uploaded, try to use the existing one
            if not (current_db_path and os.path.exists(current_db_path)):
                raise HTTPException(status_code=400, detail="No database file uploaded and no active database found. Please upload a database file.")

        try:
            table_specs = extract_schema(current_db_path)
        except sqlite3.OperationalError:
            if file:
                os.remove(current_db_path)
                current_db_path = None
            raise HTTPException(status_code=400, detail="Invalid SQLite database.")

        if not table_specs:
            if file:
                os.remove(current_db_path)
                current_db_path = None
            raise HTTPException(status_code=400, detail="No SQL tables found in the provided database.")

        rerank_table = rank_tables(query, table_specs)
        sql_prompt = make_sql_prompt(query, rerank_table)
        generated_sql = generate_sql_query(sql_prompt)

        try:
            sql_result = sql_response(generated_sql, current_db_path)
            natural_language_answer = rag_response(query, generated_sql, sql_result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SQL Execution Error: {str(e)}")

        return {
            "query": query,
            "generated_sql": generated_sql,
            "sql_result": sql_result,
            "answer": natural_language_answer
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/clear_db')
async def clear_db():
    """Deletes the currently active database."""
    global current_db_path
    if current_db_path and os.path.exists(current_db_path):
        try:
            os.remove(current_db_path)
        except Exception as e:
            print(f"Failed to delete DB: {e}")
        current_db_path = None
        return {"message": "Database cleared."}
    return {"message": "No active database found."}