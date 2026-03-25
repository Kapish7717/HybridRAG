import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

url = 'https://api.jina.ai/v1/rerank'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {JINA_API_KEY}'
}
data = {
    "model": "jina-reranker-v2-base-multilingual",
    "query": "what are the top 3 platforms by total sales",
    "documents": ["CREATE TABLE A (id INT);", "CREATE TABLE B (id INT);"],
    "top_n": 2,
    "return_documents": True
}

response = requests.post(url, headers=headers, json=data)
with open('jina_res.json', 'w') as f:
    json.dump(response.json(), f, indent=2)
