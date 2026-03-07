from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)