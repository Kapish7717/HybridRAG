import os
import pytest
from langchain_core.embeddings import Embeddings

from unittest.mock import patch

# Mock API keys so they are defined
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "mock-hf-token"

from embed import get_embeddings

def test_get_embeddings():
    embeddings = get_embeddings()
    assert embeddings is not None
    assert isinstance(embeddings, Embeddings)

@patch("langchain_huggingface.embeddings.HuggingFaceEndpointEmbeddings.embed_query")
@patch("langchain_huggingface.embeddings.HuggingFaceEndpointEmbeddings.embed_documents")
def test_embedding_functionality(mock_embed_docs, mock_embed_query):
    # Set up mock return values
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embed_docs.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    embeddings = get_embeddings()
    
    # Test single query embedding
    query_result = embeddings.embed_query("Hello world")
    assert isinstance(query_result, list)
    assert len(query_result) == 3
    assert all(isinstance(x, float) for x in query_result)
    
    # Test documents embedding
    docs_result = embeddings.embed_documents(["Hello", "World"])
    assert isinstance(docs_result, list)
    assert len(docs_result) == 2
    assert all(isinstance(doc, list) for doc in docs_result)
    assert all(isinstance(x, float) for doc in docs_result for x in doc)
