import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_query(username: str, query: str, answer: str, 
              latency_ms: float, num_chunks: int):
    """Log query for analytics"""
    log_entry = {
        "timestamp": str(datetime.now()),
        "username": username,
        "query": query,
        "answer_length": len(answer),
        "latency_ms": round(latency_ms, 2),
        "num_chunks": num_chunks,
        "success": True
    }
    logger.info(f"QUERY | {json.dumps(log_entry)}")

def log_upload(username: str, filename: str, 
               num_chunks: int, latency_ms: float):
    """Log document upload"""
    log_entry = {
        "timestamp": str(datetime.now()),
        "username": username,
        "filename": filename,
        "num_chunks": num_chunks,
        "latency_ms": round(latency_ms, 2)
    }
    logger.info(f"UPLOAD | {json.dumps(log_entry)}")

def log_error(username: str, endpoint: str, error: str):
    """Log errors"""
    log_entry = {
        "timestamp": str(datetime.now()),
        "username": username,
        "endpoint": endpoint,
        "error": error
    }
    logger.error(f"ERROR | {json.dumps(log_entry)}")