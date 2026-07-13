import os
import pytest
from langchain_core.embeddings import Embeddings

# Mock API keys so they are defined
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"

from embed import get_embeddings

def test_get_embeddings():
    embeddings = get_embeddings()
    assert embeddings is not None
    assert isinstance(embeddings, Embeddings)

def test_embedding_functionality():
    embeddings = get_embeddings()
    
    # Test single query embedding
    query_result = embeddings.embed_query("Hello world")
    assert isinstance(query_result, list)
    assert len(query_result) > 0
    assert all(isinstance(x, float) for x in query_result)
    
    # Test documents embedding
    docs_result = embeddings.embed_documents(["Hello", "World"])
    assert isinstance(docs_result, list)
    assert len(docs_result) == 2
    assert all(isinstance(doc, list) for doc in docs_result)
    assert all(isinstance(x, float) for doc in docs_result for x in doc)
