from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)