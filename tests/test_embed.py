import os

from langchain_core.embeddings import Embeddings

# Fallback keys so provider clients can be constructed in CI where no .env
# exists. Tests must never make real network calls regardless.
os.environ.setdefault("GROQ_API_KEY", "mock-groq-key")
os.environ.setdefault("GOOGLE_API_KEY", "mock-google-key")
os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", "mock-hf-token")

from embed import get_embeddings


def test_get_embeddings():
    embeddings = get_embeddings()
    assert embeddings is not None
    assert isinstance(embeddings, Embeddings)


def test_embedding_functionality(monkeypatch):
    embeddings = get_embeddings()

    # Patch the concrete provider class rather than a specific backend, so this
    # test works whether EMBED_MODEL selects Gemini, Hugging Face, etc.
    monkeypatch.setattr(
        type(embeddings),
        "embed_query",
        lambda self, text, **kwargs: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        type(embeddings),
        "embed_documents",
        lambda self, texts, **kwargs: [[0.1, 0.2, 0.3] for _ in texts],
    )

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
