from datetime import datetime


document_registry={}

def register_document(doc_id, filename, user_id=None, chunk_count=0):
    document_registry[doc_id] = {
        "filename": filename,
        "uploaded_at": datetime.now().isoformat(),
        "user_id": user_id,
        "chunk_count": chunk_count
    }

def get_user_documents(user_id):
    """Get all documents uploaded by a user"""
    return {
        doc_id: info 
        for doc_id, info in document_registry.items() 
        if info.get("user_id") == user_id
    }