import pytest
from langchain_core.documents import Document
from ingestion import calculate_chunk_ids

def test_calculate_chunk_ids_single_page():
    # Chunks on the same page should have incrementing indices
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="chunk 2", metadata={"source": "test.pdf", "page": 0}),
    ]
    
    result = calculate_chunk_ids(chunks)
    
    assert result[0].metadata["id"] == "test.pdf:0:0"
    assert result[1].metadata["id"] == "test.pdf:0:1"

def test_calculate_chunk_ids_multiple_pages():
    # Index should reset when the page changes
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="chunk 2", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="chunk 3", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="chunk 4", metadata={"source": "test.pdf", "page": 1}),
    ]
    
    result = calculate_chunk_ids(chunks)
    
    assert result[0].metadata["id"] == "test.pdf:0:0"
    assert result[1].metadata["id"] == "test.pdf:0:1"
    assert result[2].metadata["id"] == "test.pdf:1:0"
    assert result[3].metadata["id"] == "test.pdf:1:1"

def test_calculate_chunk_ids_empty():
    assert calculate_chunk_ids([]) == []
