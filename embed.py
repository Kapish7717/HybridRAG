from langchain_ollama import OllamaEmbeddings
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # ✅ ADD THIS
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL,base_url=OLLAMA_BASE_URL)