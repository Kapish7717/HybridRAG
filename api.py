from fastapi import FastAPI, UploadFile,File,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
from datetime import datetime
from ingestion import ingest_pdf
from rag import generate_answer
from typing import Dict, List, Tuple, Any,Optional
from auth import verify_token, create_user,load_users,hash_password
from rate_limiter import check_rate_limit
import time

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
def get_metrics(username: str = Depends(verify_token)):
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


@app.post("/auth/register")
def register(request:RegisterRequest):
    """"Register a new User"""
    result = create_user(request.username,request.password)
    return{
        "messsage":"user created successfully!",
        "username":result["username"],
        "token":result["token"]
    }

@app.post("/login")
def login(request:RegisterRequest):
    users=load_users()

    if request.username not in users:
        raise HTTPException(status_code=401,detail="Invalid Credentials")
    
    if users[request.username]["password"]!= hash_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return{
        "Token": users[request.username]["token"],
        "username": request.username
    }


@app.post('/ingest')
async def ingest_document(file:UploadFile=File(...),
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
        os.makedirs('uploads',exist_ok=True)
        unique_filename= f"{uuid.uuid4()}_{file.filename}"
        file_path=f"uploads/{unique_filename}"
        with open( file_path,'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        result =ingest_pdf(file_path)
        try:
            os.remove(file_path)
            print(f"🗑️ Cleaned up: {file_path}")
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
        
        

@app.post('/query',response_model=QueryResponse)
async def query_documents(request:QueryRequest,
    # username: str = Depends(verify_token)
    ):
    username="default"
    try:
        session_id = request.session_id or str(uuid.uuid4())
        session_key = f"{username}:{session_id}"
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]


        answer,sources = generate_answer(request.query,chat_history)

        chat_sessions[session_id].append((request.query,answer))

        if len(chat_sessions[session_id]) > 10:
            chat_sessions[session_id] = chat_sessions[session_id][-10:]
        

        return {
            "query": request.query,
            "answer": answer,
            "session_id":session_id,
            # "sources":sources
        }
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
@app.get('/debug/sessions/{session_id}', response_model=SessionDebugResponse)
def get_session_history(session_id: str):
    """
    Get conversation history for a specific session
    
    **Usage:** GET /debug/sessions/YOUR_SESSION_ID
    
    Example: GET /debug/sessions/1234567890
    """
    print(f"🔍 Looking for session: {session_id}")
    print(f"📊 Available sessions: {list(chat_sessions.keys())}")
    
    if session_id not in chat_sessions:
        print(f"❌ Session {session_id} not found")
        return {
            "session_id": session_id,
            "exists": False,
            "message_count": 0,
            "conversation": []
        }
    
    history = chat_sessions[session_id]
    print(f"✅ Found session with {len(history)} messages")
    
    return {
        "session_id": session_id,
        "exists": True,
        "message_count": len(history),
        "conversation": [
            {
                "turn": i + 1,
                "user_message": user_msg,
                "assistant_response": ai_msg
            }
            for i, (user_msg, ai_msg) in enumerate(history)
        ]
    }


@app.get('/debug/all-sessions')
def list_all_sessions():
    """
    List all active sessions (for debugging)
    
    **Usage:** GET /debug/all-sessions
    """
    if not chat_sessions:
        return {
            "total_sessions": 0,
            "message": "No active sessions",
            "sessions": []
        }
    
    sessions = []
    for session_id, history in chat_sessions.items():
        last_turn = history[-1] if history else (None, None)
        
        sessions.append({
            "session_id": session_id,
            "message_count": len(history),
            "last_query": last_turn[0] if last_turn[0] else None,
            "last_response": (last_turn[1][:100] + "...") if last_turn[1] and len(last_turn[1]) > 100 else last_turn[1]
        })
    
    return {
        "total_sessions": len(chat_sessions),
        "sessions": sessions
    }