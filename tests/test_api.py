import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mock API keys before importing api to avoid initial checks or dependency failures
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"

import api

@pytest.fixture
def client():
    # Clear sessions and reset state for each test
    api.chat_sessions.clear()
    api.ingestion_state = {
        "status": "idle",
        "filename": None,
        "error": None,
        "doc_id": None,
        "chunks": 0,
        "total_pages": 0
    }
    return TestClient(api.app)

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "message": "API is running"}

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "usage" in data
    assert data["system"]["status"] == "healthy"

@patch("api.generate_answer")
def test_query_endpoint(mock_generate_answer, client):
    # Setup mock return value
    mock_generate_answer.return_value = (
        "This is a mock answer.",
        [{"chunk_id": "doc.pdf:0:0", "page": "Page 0", "score": 0.95}],
        ["This is the raw context retrieved."]
    )
    
    response = client.post(
        "/query",
        json={"query": "test query", "session_id": "session123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["answer"] == "This is a mock answer."
    assert data["session_id"] == "session123"
    
    # Check that chat history is stored
    assert "session123" in api.chat_sessions
    assert len(api.chat_sessions["session123"]) == 1
    assert api.chat_sessions["session123"][0] == ("test query", "This is a mock answer.")

@patch("api.ingest_pdf")
@patch("api.refresh_bm25")
def test_ingest_endpoint_success(mock_refresh, mock_ingest, client):
    mock_ingest.return_value = {
        "doc_id": "doc_abc_123",
        "filename": "test.pdf",
        "chunks": 5,
        "total_pages": 2
    }
    
    pdf_content = b"%PDF-1.4 mock content..."
    response = client.post(
        "/ingest",
        files={"file": ("test.pdf", pdf_content, "application/pdf")}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    
    # In FastAPI TestClient, background tasks run synchronously during the request
    # so we can check final state immediately.
    assert api.ingestion_state["status"] == "success"
    assert api.ingestion_state["doc_id"] == "doc_abc_123"
    assert api.ingestion_state["chunks"] == 5
    assert api.ingestion_state["total_pages"] == 2
    
    mock_ingest.assert_called_once()
    mock_refresh.assert_called_once()

@patch("api.ingest_pdf")
def test_ingest_endpoint_failure(mock_ingest, client):
    mock_ingest.side_effect = Exception("Failed to read PDF pages")
    
    pdf_content = b"%PDF-1.4 mock content..."
    response = client.post(
        "/ingest",
        files={"file": ("test.pdf", pdf_content, "application/pdf")}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    
    # Check that the ingestion state has marked it as failed
    assert api.ingestion_state["status"] == "failed"
    assert api.ingestion_state["error"] == "Failed to read PDF pages"

def test_clear_db_no_active(client):
    api.current_db_path = None
    response = client.post("/clear_db")
    assert response.status_code == 200
    assert response.json() == {"message": "No active database found."}

@patch("os.path.exists")
@patch("os.remove")
def test_clear_db_active(mock_remove, mock_exists, client):
    mock_exists.return_value = True
    api.current_db_path = "/path/to/db.db"
    
    response = client.post("/clear_db")
    assert response.status_code == 200
    assert response.json() == {"message": "Database cleared."}
    assert api.current_db_path is None
    mock_remove.assert_called_once_with("/path/to/db.db")
