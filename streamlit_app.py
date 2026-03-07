import streamlit as st
import requests
import uuid

import os

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat Assistant", page_icon="📚")

st.title("📚 RAG Chat Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    if response.status_code == 200:
                        st.success(f"Successfully processed {uploaded_file.name}")
                        st.session_state.document_uploaded = True
                        st.session_state.messages = [] # clear chat
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages.append({"role": "assistant", "content": f"Document '{uploaded_file.name}' is ready. Ask me anything!"})
                    else:
                        st.error(f"Failed to process: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
        else:
            st.warning("Please select a file first.")
            
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the document" if st.session_state.document_uploaded else "Please upload a document first", disabled=not st.session_state.document_uploaded):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_BASE_URL}/query", json={
                    "query": prompt,
                    "session_id": st.session_state.session_id
                })
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    if "session_id" in data:
                        st.session_state.session_id = data["session_id"]
                else:
                    error_msg = f"Error: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error connecting to backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
