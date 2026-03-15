import streamlit as st
import requests
import uuid

import os

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat Assistant", page_icon="📚")

st.title("📚 RAG Chat Assistant")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "db_uploaded" not in st.session_state:
    st.session_state.db_uploaded = False

with st.sidebar:
    st.header("Upload Data source")
    
    # Let user select the mode
    mode = st.radio("Select Mode:", ("PDF Document (RAG)", "SQLite Database (Text-to-SQL)"))
    st.session_state.mode = mode
    
    if mode == "PDF Document (RAG)":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if st.button("Upload PDF"):
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
                
    elif mode == "SQLite Database (Text-to-SQL)":
        uploaded_db = st.file_uploader("Choose a SQLite Database", type=["db", "sqlite", "sqlite3"])
        
        # In SQL mode, we send the file directly with the query to the /sql_query endpoint 
        # But we still want a setup/clear button to mark the state
        if st.button("Set Database"):
            if uploaded_db is not None:
                st.session_state.uploaded_db = uploaded_db
                st.session_state.db_uploaded = True
                st.session_state.messages = [] # clear chat
                st.session_state.messages.append({"role": "assistant", "content": f"Database '{uploaded_db.name}' is set. Ask me a question and I'll generate the SQL!"})
                st.success("Database set for querying!")
            else:
                st.warning("Please select a database file first.")
                
        if st.button("Clear Active Database"):
            with st.spinner("Clearing Database..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/clear_db")
                    if res.status_code == 200:
                        st.session_state.db_uploaded = False
                        st.session_state.uploaded_db = None
                        st.session_state.messages = []
                        st.success("Database cleared!")
                except Exception as e:
                     st.error(f"Error: {e}")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Determine if the user is allowed to chat based on the selected mode
is_ready = False
placeholder_text = ""
if st.session_state.mode == "PDF Document (RAG)":
    is_ready = st.session_state.document_uploaded
    placeholder_text = "Ask a question about the document" if is_ready else "Please upload a document first"
else:
    is_ready = st.session_state.db_uploaded
    placeholder_text = "Ask a question about the database" if is_ready else "Please upload a database first"

if prompt := st.chat_input(placeholder_text, disabled=not is_ready):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..." if st.session_state.mode == "PDF Document (RAG)" else "Generating SQL & Querying..."):
            try:
                if st.session_state.mode == "PDF Document (RAG)":
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
                
                elif st.session_state.mode == "SQLite Database (Text-to-SQL)":
                    # For SQL mode, we send the file and the query as Form data
                    db_file = st.session_state.get("uploaded_db")
                    
                    files = None
                    if db_file:
                        # We send the file to update the global state in the backend.
                        files = {"file": (db_file.name, db_file.getvalue(), "application/octet-stream")}
                        
                    data = {"query": prompt}
                    
                    response = requests.post(f"{API_BASE_URL}/sql_query", data=data, files=files)
                    
                    if response.status_code == 200:
                        resp_data = response.json()
                        sql_query = resp_data.get("generated_sql", "")
                        sql_result = resp_data.get("sql_result", [])
                        nl_answer = resp_data.get("answer", "No natural language answer returned.")
                        
                        # Format the output beautifully
                        answer_md = f"**Answer:** {nl_answer}\n\n"
                        answer_md += f"<details><summary>View Details</summary>\n\n**Generated SQL:**\n```sql\n{sql_query}\n```\n\n**Raw Result:**\n`{sql_result}`\n</details>"
                        
                        st.markdown(answer_md, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": answer_md})
                        
                        # Once we've sent the file once and it's active in the backend, we technically 
                        # don't need to send it again (we can save bandwidth). Let's clear the uploaded_db state.
                        if st.session_state.get("uploaded_db"):
                           st.session_state.uploaded_db = None 
                    else:
                        error_msg = f"Error: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
            except Exception as e:
                error_msg = f"Error connecting to backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
