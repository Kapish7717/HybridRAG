from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import add_documents, clear_vector_store, get_all_ids
import os
from datetime import datetime
import uuid
import time

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  
LLM_MODEL = os.getenv("LLM_MODEL", "phi3:mini")
TOP_K = int(os.getenv("TOP_K", "4"))




def ingest_pdf(pdf_path:str,username: str = "default"):
    # Generate unique document ID
    doc_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    filename = os.path.basename(pdf_path)
    
    print(f"📄 Processing: {filename} (doc_id: {doc_id})")

    # The app keeps a single active document, so clear the namespace first.
    try:
        cleared = clear_vector_store()
        if cleared:
            print(f"🗑️ Clearing {cleared} old vectors from Pinecone...")
        print("✅ Vector index cleared.")
    except Exception as e:
        print(f"⚠️ Could not clear Pinecone index: {e}")
        print("Proceeding with existing index (chunks will be added/updated as needed).")

    document_loader = PyPDFLoader(pdf_path)
    documents = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]



    chunks_with_ids = calculate_chunk_ids(chunks)

    try:
        existing_ids = set(get_all_ids())
    except Exception as e:
        print(f"⚠️ Could not fetch existing vector ids: {e}")
        existing_ids = set()
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        # Batch insertions
        batch_size = 50
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = new_chunk_ids[i:i + batch_size]
            print(f"Inserting batch {i//batch_size + 1}/{(len(new_chunks) + batch_size - 1)//batch_size}...")
            add_documents(batch, ids=batch_ids, batch_size=batch_size)

            # Be gentle with hosted embedding APIs (e.g. Gemini free tier).
            if i + batch_size < len(new_chunks):
                print("Sleeping for 10 seconds to respect embedding API rate limits...")
                time.sleep(10)

        print("✅ Successfully added to database")
    else:
        print("✅ No new documents to add")
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "new_chunks": len(new_chunks),
        "total_pages": len(documents)
    }


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks