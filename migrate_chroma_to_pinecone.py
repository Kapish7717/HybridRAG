"""Copy an existing ChromaDB collection into a Pinecone index.

The vectors are copied **as-is** (no re-embedding), so the target Pinecone index
must use the same embedding dimension as the source collection. The page text is
stored in the ``text`` metadata field, which is what ``langchain-pinecone`` reads
back, and the original Chroma id is kept in the ``id`` metadata field.

Usage
-----
    # Inspect the source collection without writing anything
    python migrate_chroma_to_pinecone.py --chroma-path ../Ollama_RAG/chroma --dry-run

    # Copy into the index configured in .env
    python migrate_chroma_to_pinecone.py --chroma-path ../Ollama_RAG/chroma

    # Recreate the index first (deletes all existing vectors)
    python migrate_chroma_to_pinecone.py --chroma-path ../Ollama_RAG/chroma --recreate

Environment
-----------
PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
PINECONE_NAMESPACE, PINECONE_METRIC (see .env).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from vector_store import TEXT_KEY, _is_ready, sanitize_metadata, safe_vector_id

load_dotenv()

# Windows consoles default to cp1252, which cannot print the status emojis.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BATCH_SIZE = 100
VECTOR_TUPLE = Tuple[str, List[float], Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate a ChromaDB collection into a Pinecone index."
    )
    parser.add_argument(
        "--chroma-path",
        default=os.getenv("CHROMA_PATH", "./chroma"),
        help="Path to the Chroma persist directory (default: %(default)s).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name. Defaults to the only/first collection.",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List the collections in the Chroma directory and exit.",
    )
    parser.add_argument(
        "--index-name",
        default=os.getenv("PINECONE_INDEX_NAME", "hybrid-rag"),
        help="Pinecone index name (default: %(default)s).",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", ""),
        help="Pinecone namespace (default: the default namespace).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Vectors per upsert request (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read Chroma and print the plan without touching Pinecone.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Pinecone index before copying (DESTRUCTIVE).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for --recreate.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
def open_chroma(path: str):
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "chromadb is not installed. Run: pip install chromadb==1.4.1"
        ) from exc

    if not os.path.isdir(path):
        raise SystemExit(f"Chroma directory not found: {path}")

    return chromadb.PersistentClient(path=path)


def collection_names(client) -> List[str]:
    names = []
    for collection in client.list_collections():
        names.append(collection if isinstance(collection, str) else collection.name)
    return names


def pick_collection(client, name: Optional[str]):
    names = collection_names(client)
    if not names:
        raise SystemExit("No collections found in the Chroma directory.")

    if name is None:
        name = "langchain" if "langchain" in names else names[0]

    if name not in names:
        raise SystemExit(
            f"Collection '{name}' not found. Available collections: {', '.join(names)}"
        )

    return client.get_collection(name)


def read_chroma(collection, page_size: int = 1000):
    """Return (ids, documents, metadatas, embeddings) from a collection."""
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page_size,
            offset=offset,
        )
        batch_ids = batch.get("ids") or []
        if not batch_ids:
            break

        batch_docs = batch.get("documents") or [None] * len(batch_ids)
        batch_metas = batch.get("metadatas") or [None] * len(batch_ids)
        batch_embs = batch.get("embeddings")
        if batch_embs is None:
            raise SystemExit(
                "Chroma returned no embeddings. Re-run with a collection that "
                "stores vectors, or re-embed the documents before migrating."
            )

        for doc_id, doc, meta, emb in zip(
            batch_ids, batch_docs, batch_metas, batch_embs
        ):
            ids.append(str(doc_id))
            documents.append(doc or "")
            metadatas.append(meta or {})
            embeddings.append(
                [float(value) for value in (emb.tolist() if hasattr(emb, "tolist") else emb)]
            )

        offset += len(batch_ids)
        print(f"  read {offset} vectors from Chroma...")

    return ids, documents, metadatas, embeddings


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
def wait_until_ready(pc: Pinecone, name: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_ready(pc.describe_index(name)):
            return
        time.sleep(2)
    raise SystemExit(f"Pinecone index '{name}' was not ready after {timeout}s.")


def wait_until_deleted(pc: Pinecone, name: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pc.has_index(name):
            return
        time.sleep(2)
    raise SystemExit(f"Pinecone index '{name}' was not deleted after {timeout}s.")


def prepare_index(
    pc: Pinecone,
    name: str,
    dimension: int,
    recreate: bool,
) -> None:
    metric = os.getenv("PINECONE_METRIC", "cosine")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    if recreate and pc.has_index(name):
        print(f"🗑️  Deleting existing index '{name}'...")
        pc.delete_index(name)
        wait_until_deleted(pc, name)

    if pc.has_index(name):
        info = pc.describe_index(name)
        existing_dimension = int(getattr(info, "dimension", 0) or 0)
        if existing_dimension != dimension:
            raise SystemExit(
                f"Index '{name}' has dimension {existing_dimension}, but the "
                f"Chroma collection has dimension {dimension}. Re-run with "
                f"--recreate or choose another index."
            )
        print(
            f"✅ Using existing index '{name}' "
            f"(dimension={existing_dimension}, metric={getattr(info, 'metric', metric)})"
        )
        wait_until_ready(pc, name)
        return

    print(
        f"🌲 Creating index '{name}' (dimension={dimension}, metric={metric}, "
        f"cloud={cloud}, region={region})..."
    )
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    wait_until_ready(pc, name)
    print(f"✅ Index '{name}' is ready.")


def build_vectors(
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> List[VECTOR_TUPLE]:
    vectors: List[VECTOR_TUPLE] = []
    for doc_id, text, metadata, embedding in zip(
        ids, documents, metadatas, embeddings
    ):
        clean = sanitize_metadata(metadata)
        clean["id"] = doc_id
        clean[TEXT_KEY] = text
        vectors.append((safe_vector_id(doc_id), embedding, clean))
    return vectors


def upsert_vectors(
    index,
    vectors: List[VECTOR_TUPLE],
    namespace: Optional[str],
    batch_size: int,
) -> None:
    total = len(vectors)
    for start in range(0, total, batch_size):
        index.upsert(
            vectors=vectors[start : start + batch_size],
            namespace=namespace or None,
        )
        print(f"  upserted {min(start + batch_size, total)}/{total}")


def count_vectors(index, namespace: Optional[str]) -> int:
    stats = index.describe_index_stats()
    if namespace:
        namespace_stats = (stats.namespaces or {}).get(namespace)
        return int(getattr(namespace_stats, "vector_count", 0) or 0)
    return int(getattr(stats, "total_vector_count", 0) or 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    print(f"📂 Opening Chroma at {os.path.abspath(args.chroma_path)}")
    client = open_chroma(args.chroma_path)

    if args.list_collections:
        for name in collection_names(client):
            print(f"  - {name}")
        return 0

    collection = pick_collection(client, args.collection)
    print(f"📚 Reading collection '{collection.name}'...")
    ids, documents, metadatas, embeddings = read_chroma(collection)

    if not ids:
        print("⚠️  The Chroma collection is empty. Nothing to migrate.")
        return 0

    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise SystemExit(f"Inconsistent embedding dimensions: {sorted(dimensions)}")
    dimension = dimensions.pop()

    print(
        f"✅ Read {len(ids)} vectors from Chroma "
        f"(dimension={dimension}, collection='{collection.name}')"
    )

    if args.dry_run:
        print("\n🧪 Dry run - no data was written to Pinecone.")
        print(f"   Target index : {args.index_name}")
        print(f"   Target namespace: {args.namespace or '(default)'}")
        print(f"   Vectors      : {len(ids)}")
        return 0

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit(
            "PINECONE_API_KEY is not set. Add it to .env or export it first."
        )

    pc = Pinecone(api_key=api_key)

    if args.recreate and not args.yes:
        answer = input(
            f"⚠️  This will DELETE index '{args.index_name}' and all its vectors. "
            "Continue? [y/N] "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    prepare_index(pc, args.index_name, dimension, recreate=args.recreate)
    index = pc.Index(args.index_name)

    vectors = build_vectors(ids, documents, metadatas, embeddings)
    upsert_vectors(
        index,
        vectors,
        namespace=args.namespace,
        batch_size=max(1, args.batch_size),
    )

    stored = count_vectors(index, args.namespace or None)
    print(f"\n📊 Pinecone now reports {stored} vectors in the target namespace.")

    sample_ids = [vector[0] for vector in vectors[:3]]
    sample = index.fetch(ids=sample_ids, namespace=args.namespace or None)
    missing_text = [
        vector_id
        for vector_id, value in sample.vectors.items()
        if not (value.metadata or {}).get(TEXT_KEY)
    ]
    if missing_text:
        print(f"⚠️  Sample vectors missing '{TEXT_KEY}' metadata: {missing_text}")
    else:
        print(f"✅ Verified sample vectors (text + metadata present).")

    if stored < len(ids):
        print(
            f"⚠️  Expected at least {len(ids)} vectors but Pinecone reports {stored}. "
            "The index may still be indexing; re-check in a moment."
        )
        return 2

    print("🎉 Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
