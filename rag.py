from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import os
from embed import get_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage 
from sentence_transformers import CrossEncoder
from hallucination_check import check_hallucination
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K", "5"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CROSS_ENCODER = os.getenv("CROSS_ENCODER","cross-encoder/ms-marco-TinyBERT-L-2-v2")



     

# def get_vectordb():
#     global _vector_db
#     if _vector_db is None:
#         embeddings = OllamaEmbeddings(model=EMBED_MODEL,base_url=OLLAMA_BASE_URL)
#         _vector_db = Chroma(
#             persist_directory=CHROMA_PATH,
#             embedding_function=embeddings
#             )
#     return _vector_db
def retrieve_context(query: str,use_hybrid=True):
    print("🔥 File loaded")

    """Retrieve top-k relevant chunks from Chroma"""
    try:  # ✅ ADD TRY BLOCK
        db = Chroma(
             persist_directory=CHROMA_PATH,embedding_function=get_embeddings()
        )
        if not use_hybrid:
            # Vector search only
            results = db.similarity_search_with_score(query, k=TOP_K)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            print("only vector search occured")
            return context_text, results
        vector_results = db.similarity_search_with_score(query, k=TOP_K)
        vector_docs = [doc for doc, score in vector_results]
        # print(len(vector_docs))

        all_data = db.get()

        if not all_data or not all_data.get("documents"):
            print("⚠️ Chroma DB is empty, using vector search only")
            context_text = "\n\n---\n\n".join([doc.page_content for doc in vector_docs])
            return context_text, vector_results
        
        all_docs = [
             Document(page_content=text,metadata=meta)
             for text, meta in zip(all_data["documents"],all_data["metadatas"])
        ]

        bm25_retriever = BM25Retriever.from_documents(all_docs,k=TOP_K)
        bm25_docs = bm25_retriever.invoke(query)

        combined_docs = vector_docs + bm25_docs

        seen=set()

        unique_docs=[]

        for doc in combined_docs:
             content = doc.page_content.strip()
             if content not in seen:
                  seen.add(content)
                  unique_docs.append(doc)

        unique_docs = unique_docs[:TOP_K]

        reranker = CrossEncoder(CROSS_ENCODER)
        print(f"🎯 Reranking to top {TOP_K}...")
        pairs = [(query, doc.page_content) for doc in unique_docs]
        scores = reranker.predict(pairs)

        scored_docs = list(zip(unique_docs,scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_results = [doc for doc, score in scored_docs[:TOP_K]]
        reranked_scores = [score for doc, score in scored_docs[:TOP_K]]

        print("   Reranking scores:")
        for i, (doc, score) in enumerate(zip(reranked_results, reranked_scores)):
            print(f"   {i+1}. Score: {score:.3f} - {doc.page_content[:60]}...")
    
        
        # 6. Format context
        context_text = "\n\n---\n\n".join([
        f"[Source {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(reranked_results)
        ])
        
        # 7. Create results in same format as vector search
        # results = [(doc, 1.0) for doc in unique_docs]  # Dummy scores after hybrid
        # print(f"Context text type: {type(context_text)}\n Results type: {type(results)}")
        # print(results[0])
        return context_text, scored_docs
    
    except Exception as e:  
        print(f"❌ Context retrieval failed: {str(e)}")
        raise

def format_chat_history(chat_history):
    """
    Format chat history into a readable string for the prompt
    
    Args:
        chat_history: List of (user_msg, ai_msg) tuples
    
    Returns:
        Formatted string
    """
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for user_msg, ai_msg in chat_history:
        formatted.append(f"User: {user_msg}")
        formatted.append(f"Assistant: {ai_msg}")
    
    return "\n".join(formatted)




def generate_answer(query:str,chat_history=None):
    """
    Generate answer with chat history support
    
    Args:
        query: Current user question
        chat_history: List of (user_msg, ai_msg) tuples from previous turns
                     Example: [("What is ML?", "Machine learning is..."), 
                              ("Give examples", "Examples include...")]
    
    Returns:
        AI response text
    """
    if chat_history is None:
        chat_history = []
    print("🔍 Retrieving context...")
    context, results = retrieve_context(query)

    history_text =  format_chat_history(chat_history)


        
    prompt = f"""You are a helpful AI assistant. Use the context and conversation history to answer the question.

CONVERSATION HISTORY:
{history_text}

CONTEXT FROM DOCUMENTS:
{context}

CURRENT QUESTION:
{query}

INSTRUCTIONS:
- If the question refers to previous conversation (e.g., "it", "that", "explain more"), use the conversation history
- Always prioritize information from the CONTEXT when answering
- If the answer is not in the context or history, say you don't know
- Keep your response concise and relevant

ANSWER:
"""

        # print("🧠 Calling Ollama LLM...")
        # client = ollama.Client(host=OLLAMA_BASE_URL)
        # response = client.chat(
        #     model=LLM_MODEL,
        #     messages=[
        #         {"role": "user", "content": prompt},
        #     ],
        # )

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file!")
        
    model = ChatGroq(model_name=LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.1)
    
    # Langchain Chat Models need a slightly different stream processing for Document/Role metadata compared to raw LLMs occasionally
    response = model.invoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
            
    # Safely unpack results which might be Document objects directly or tuples of (Document, score)
    sources_raw = []
    raw_contexts = []
    
    for item in results:
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        elif isinstance(item, Document):
            doc = item
            score = 1.0 # default score
        else:
            continue
            
        sources_raw.append((doc, score))
        raw_contexts.append(doc.page_content)

    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    sources=[]
    for doc,score in sources_raw:
        chunk_id = doc.metadata.get("id", "unknown")
        
        # Parse chunk_id: "uploads/doc.pdf:5:2" -> page 5
        if ":" in chunk_id:
            parts = chunk_id.split(":")
            if len(parts) >= 2:
                page_num = parts[1]
                source_info = f"Page {page_num}"
            else:
                source_info = chunk_id
        else:
            source_info = chunk_id
        
        sources.append({
            "chunk_id": chunk_id,
            "page": source_info,
            "score": float(score)
        })
    
    # print(f"Sources type = {type(sources)}")
    # print(formatted_response)
    # Since we are returning a stream, we can't synchronously check hallucination here easily. 
    # We will defer hallucination and Ragas checks to the background task in api.py.
    return response_text, sources, raw_contexts
# if __name__ =='__main__':
#      query = "what is this document about?"
#      retrieve_context(query,True)