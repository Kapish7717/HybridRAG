import json
import os
from datetime import datetime, timedelta

SESSION_FILE = "sessions.json"


def load_sessions():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    with open(SESSION_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def get_session(session_id: str, username: str) -> list:
    """Get chat history for a session"""
    sessions = load_sessions()
    key = f"{username}:{session_id}"
    
    if key not in sessions:
        return []
    
    return sessions[key]["history"]

def update_session(session_id: str, username: str, query: str, answer: str):
    """Add message to session"""
    sessions = load_sessions()
    key = f"{username}:{session_id}"
    
    if key not in sessions:
        sessions[key] = {
            "username": username,
            "session_id": session_id,
            "created_at": str(datetime.now()),
            "history": []
        }
    
    sessions[key]["history"].append({
        "query": query,
        "answer": answer,
        "timestamp": str(datetime.now())
    })
    
    # Keep last 10
    if len(sessions[key]["history"]) > 10:
        sessions[key]["history"] = sessions[key]["history"][-10:]
    
    sessions[key]["updated_at"] = str(datetime.now())
    save_sessions(sessions)


def get_user_sessions(username: str) -> list:
    """Get all sessions for a user"""
    sessions = load_sessions()
    user_sessions = []
    
    for key, data in sessions.items():
        if key.startswith(f"{username}:"):
            user_sessions.append({
                "session_id": data["session_id"],
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data["history"])
            })
    
    return user_sessions