import os

os.environ.setdefault("PINECONE_API_KEY", "mock-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

from vector_store import safe_vector_id, sanitize_metadata


def test_sanitize_metadata_drops_none_and_internal_keys():
    metadata = {
        "id": "test.pdf:0:1",
        "page": 0,
        "title": None,
        "chroma:document": "internal text",
        "text": "duplicated text",
        "tags": ["a", "b"],
    }

    assert sanitize_metadata(metadata) == {
        "id": "test.pdf:0:1",
        "page": 0,
        "tags": ["a", "b"],
    }


def test_sanitize_metadata_serializes_nested_values():
    cleaned = sanitize_metadata({"nested": {"a": 1}, "numbers": [1, 2]})

    assert cleaned["nested"] == '{"a": 1}'
    assert cleaned["numbers"] == "[1, 2]"


def test_sanitize_metadata_handles_empty_metadata():
    assert sanitize_metadata(None) == {}
    assert sanitize_metadata({}) == {}


def test_safe_vector_id_preserves_ascii_ids():
    assert safe_vector_id("test.pdf:0:1") == "test.pdf:0:1"


def test_safe_vector_id_hashes_unsafe_ids():
    long_id = "x" * 600
    hashed = safe_vector_id(long_id)

    assert hashed != long_id
    assert len(hashed) == 40
    assert hashed.isalnum()
