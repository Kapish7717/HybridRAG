"""Pinecone vector store integration.

This module replaces the previous ChromaDB backend. It exposes the small set of
operations the RAG pipeline needs:

* ``similarity_search_with_score`` - dense retrieval
* ``get_all_documents``           - full read used to rebuild the BM25 index
* ``add_documents``               - batched upserts during ingestion
* ``clear_vector_store``          - wipe the namespace before a new upload

The index is created automatically (serverless) if it does not exist yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from embed import get_embeddings

load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "3072"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_INDEX_TIMEOUT = int(os.getenv("PINECONE_INDEX_TIMEOUT", "180"))

# PineconeVectorStore stores the page content inside metadata under this key.
TEXT_KEY = "text"
FETCH_BATCH_SIZE = 100  # Pinecone allows up to 1000 ids per fetch request.

# Global singletons (the API process reuses one client/index/store).
_pc: Optional[Pinecone] = None
_index: Optional[Any] = None
_vector_store: Optional[PineconeVectorStore] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_api_key() -> str:
    api_key = os.getenv("PINECONE_API_KEY") or PINECONE_API_KEY
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file or environment."
        )
    return api_key


def _get_index_name() -> str:
    return os.getenv("PINECONE_INDEX_NAME") or PINECONE_INDEX_NAME


def _get_namespace(namespace: Optional[str] = None) -> Optional[str]:
    ns = PINECONE_NAMESPACE if namespace is None else namespace
    return ns or None


def sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return metadata that Pinecone can store.

    Pinecone only accepts flat metadata whose values are strings, numbers,
    booleans or lists of strings. Chroma internals (``chroma:*`` keys) and the
    page-content key are stripped because the text is stored separately.
    """
    clean: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue

        key = str(key)
        if key.startswith("chroma:") or key == TEXT_KEY:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            if all(isinstance(item, str) for item in value):
                clean[key] = list(value)
            else:
                clean[key] = json.dumps(value, default=str)
        else:
            clean[key] = json.dumps(value, default=str)

    return clean


def safe_vector_id(value: str) -> str:
    """Pinecone ids must be ASCII, printable and at most 512 chars long."""
    value = str(value)
    if len(value) <= 512 and value.isascii() and all(ord(c) >= 32 for c in value):
        return value
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _is_ready(index_info: Any) -> bool:
    status = getattr(index_info, "status", None)
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


# ---------------------------------------------------------------------------
# Client / index management
# ---------------------------------------------------------------------------
def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=_get_api_key())
    return _pc


def get_index():
    """Return a handle to the configured Pinecone index."""
    global _index
    if _index is None:
        ensure_index()
        _index = get_pinecone_client().Index(_get_index_name())
    return _index


def ensure_index(dimension: Optional[int] = None):
    """Create the index if needed and wait until it is ready."""
    pc = get_pinecone_client()
    name = _get_index_name()

    if pc.has_index(name):
        info = pc.describe_index(name)
        index_dimension = getattr(info, "dimension", None)
        if dimension and index_dimension and int(index_dimension) != int(dimension):
            raise ValueError(
                f"Pinecone index '{name}' has dimension {index_dimension}, "
                f"but the embeddings produce {dimension} dimensions."
            )
        return info

    dimension = int(dimension or PINECONE_DIMENSION)
    print(
        f"🌲 Creating Pinecone index '{name}' "
        f"(dimension={dimension}, metric={PINECONE_METRIC}, "
        f"cloud={PINECONE_CLOUD}, region={PINECONE_REGION})..."
    )
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    deadline = time.time() + PINECONE_INDEX_TIMEOUT
    while time.time() < deadline:
        info = pc.describe_index(name)
        if _is_ready(info):
            print(f"✅ Pinecone index '{name}' is ready.")
            return info
        time.sleep(2)

    raise TimeoutError(
        f"Pinecone index '{name}' was not ready after {PINECONE_INDEX_TIMEOUT}s."
    )


def get_vector_store(force_new: bool = False) -> PineconeVectorStore:
    """Return a LangChain ``PineconeVectorStore`` backed by the index."""
    global _vector_store
    if _vector_store is None or force_new:
        ensure_index()
        _vector_store = PineconeVectorStore(
            index=get_index(),
            embedding=get_embeddings(),
            text_key=TEXT_KEY,
            namespace=_get_namespace(),
        )
    return _vector_store


# ---------------------------------------------------------------------------
# Read / write operations
# ---------------------------------------------------------------------------
def add_documents(
    documents: List[Document],
    ids: Optional[List[str]] = None,
    batch_size: int = 50,
    namespace: Optional[str] = None,
) -> List[str]:
    """Embed and upsert documents into Pinecone.

    ``ids`` are the logical chunk ids (stored in metadata as ``id``). The
    Pinecone vector id is derived from them with :func:`safe_vector_id`.
    """
    if not documents:
        return []

    store = get_vector_store()
    texts = [doc.page_content for doc in documents]
    metadatas = [sanitize_metadata(doc.metadata) for doc in documents]

    if ids is None:
        ids = [
            metadata.get("id") or f"chunk-{uuid.uuid4().hex}"
            for metadata in metadatas
        ]

    vector_ids = [safe_vector_id(chunk_id) for chunk_id in ids]

    return store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=vector_ids,
        namespace=_get_namespace(namespace),
        batch_size=batch_size,
    )


def similarity_search_with_score(
    query: str,
    k: int = 4,
    namespace: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    return get_vector_store().similarity_search_with_score(
        query, k=k, namespace=_get_namespace(namespace)
    )


def get_all_ids(namespace: Optional[str] = None) -> List[str]:
    """List every vector id in the namespace."""
    index = get_index()
    ids: List[str] = []
    for page in index.list(namespace=_get_namespace(namespace)):
        ids.extend(page)
    return ids


def get_all_documents(namespace: Optional[str] = None) -> List[Document]:
    """Fetch every stored chunk (used to rebuild the in-memory BM25 index)."""
    index = get_index()
    ns = _get_namespace(namespace)
    ids = get_all_ids(namespace=ns)

    documents: List[Document] = []
    for start in range(0, len(ids), FETCH_BATCH_SIZE):
        batch = ids[start : start + FETCH_BATCH_SIZE]
        response = index.fetch(ids=batch, namespace=ns)
        for vector_id, vector in response.vectors.items():
            metadata = dict(vector.metadata or {})
            text = metadata.pop(TEXT_KEY, "") or ""
            if not text.strip():
                continue
            metadata.setdefault("id", vector_id)
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def count_vectors(namespace: Optional[str] = None) -> int:
    stats = get_index().describe_index_stats()
    ns = _get_namespace(namespace)
    if ns:
        namespace_stats = (stats.namespaces or {}).get(ns)
        return int(getattr(namespace_stats, "vector_count", 0) or 0)
    return int(getattr(stats, "total_vector_count", 0) or 0)


def clear_vector_store(namespace: Optional[str] = None) -> int:
    """Delete every vector in the namespace. Returns the previous count."""
    index = get_index()
    ns = _get_namespace(namespace)
    previous = count_vectors(namespace=ns)
    if previous:
        index.delete(delete_all=True, namespace=ns)
    return previous


def get_index_status() -> Dict[str, Any]:
    pc = get_pinecone_client()
    name = _get_index_name()
    info = pc.describe_index(name)
    return {
        "name": name,
        "dimension": getattr(info, "dimension", None),
        "metric": getattr(info, "metric", None),
        "host": getattr(info, "host", None),
        "ready": _is_ready(info),
        "namespace": _get_namespace(),
        "vector_count": count_vectors(),
    }
