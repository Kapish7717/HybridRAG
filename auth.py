from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from typing import Annotated
import hashlib
import os
import json
import secrets
from datetime import datetime
security = HTTPBearer()

USER_FILE  = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE,'r') as f:
            return json.load(f)
    
    return {}

def save_users(users):
    with open(USER_FILE,'w')as f:
        json.dump(users,f)

def hash_password(password:str)->str:
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token()->str:
    return secrets.token_urlsafe(32)

def create_user(username:str,password:str)->dict:
    users=load_users()

    if username in users:
        raise HTTPException(status_code=400,detail="User already exists!")
    
    token = generate_token()
    users[username]={
        "password": hash_password(password),
        "token": token,
        "created_at": str(datetime.now()),
        "uploads": [],
        "query_count": 0
    } 
    save_users(users)
    return {
        "username":username,
        "token":token
    }


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    users = load_users()
    token =credentials.credentials
    for username, data in users.items():
        if data["token"]==token:
            return username
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invaid or Expired token"
    )

        