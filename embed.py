from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

class RateLimitedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        # Enforce a safe chunk/batch size for the Gemini Embeddings API (e.g., 8 texts per API request)
        batch_size = kwargs.pop('batch_size', 8)
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Embed this batch with robust exponential backoff retry logic
            batch_result = None
            max_retries = 6
            backoff = 2
            for attempt in range(max_retries):
                try:
                    # Call the superclass method for just this batch
                    batch_result = super().embed_documents(batch, batch_size=batch_size, **kwargs)
                    break
                except Exception as e:
                    print(f"[WARNING] Rate limited or error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    # Exponential backoff with jitter
                    sleep_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                    print(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                    time.sleep(sleep_time)
            
            if batch_result:
                results.extend(batch_result)
                
            # If there are more batches, sleep slightly to prevent exceeding the Requests Per Minute (RPM) limit
            if i + batch_size < len(texts):
                time.sleep(1.5)
                
        return results

    def embed_query(self, text: str, **kwargs) -> list[float]:
        max_retries = 6
        backoff = 2
        for attempt in range(max_retries):
            try:
                return super().embed_query(text, **kwargs)
            except Exception as e:
                print(f"[WARNING] Rate limited or error on query embedding attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise e
                sleep_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                print(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)

def get_embeddings():
    return RateLimitedGoogleEmbeddings(model=EMBED_MODEL)