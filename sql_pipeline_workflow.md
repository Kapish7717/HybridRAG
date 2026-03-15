# Text-to-SQL Pipeline Workflow

This document explains the step-by-step workflow of the newly integrated **Text-to-SQL** feature, which allows users to upload an SQLite database and interact with it using natural language.

---

## 1. Upload & Initialization
- **Action:** The user uploads a valid SQLite database file (`.db`, `.sqlite`) via the Streamlit interface.
- **Backend Process:**
  - The FastAPI backend receives the file via the `/sql_query` endpoint.
  - It saves the database to a secure, hidden **temporary file** (`tempfile.NamedTemporaryFile`).
  - The path to this active temporary database is stored centrally in the backend memory (`current_db_path`), allowing the user to make repeated queries without needing to re-upload the file every time.

## 2. Schema Extraction
- **Action:** The API extracts the raw structural schema from the newly uploaded database.
- **Backend Process:**
  - The system connects to the temporary database and queries the internal `sqlite_master` table.
  - It fetches the exact `CREATE TABLE` statements for every user-defined table in the database, ignoring internal generic tables (like `sqlite_sequence`).
  - These schema definitions are stored in memory as a list of strings (`table_specs`).

## 3. Semantic Table Reranking
- **Action:** The user submits a natural language question (e.g., *"What were the total sales in the West region?"*).
- **Backend Process:**
  - Standard databases can have dozens or hundreds of tables, which would easily overwhelm an LLM's context window.
  - The system pairs the user's query with every single table schema extracted in Step 2.
  - Using a Cross-Encoder (`jinaai/jina-reranker-v2-base-multilingual`), the system mathematically scores how statistically relevant each table is to the user's question.
  - The tables are ranked, and only the mathematically **Top 3 most relevant tables** are sent forward.

## 4. SQL Generation
- **Action:** A prompt is constructed containing the user's question and the Top 3 table schemas.
- **Backend Process:**
  - The system utilizes **Groq API** (powered by `llama-3.3-70b-versatile`) to generate a highly precise SQLite query.
  - The prompt explicitly forces the LLM to output *only* raw SQL code with no conversational filler, ensuring it can be executed mechanically.

## 5. Execution & Data Retrieval
- **Action:** The generated SQL is executed securely against the user's active temporary database.
- **Backend Process:**
  - The server cleans the LLM response of any markdown backticks.
  - It opens a connection to `current_db_path` and executes the query through a standard cursor.
  - The raw database response (usually a list of tuples) is captured via `.fetchall()`.

## 6. Conversational Response Generation (RAG)
- **Action:** The system translates the raw database tuples back into a human-friendly answer.
- **Backend Process:**
  - The system constructs a final prompt consisting of:
    1. The user's original natural language question.
    2. The exact SQL query that was generated.
    3. The JSON representation of the raw database result (`sql_result`).
  - The Groq LLM reads this structured data and answers the user naturally (e.g., *"The total sales in the West region were $45,000."*).

## 7. Frontend Display & Cleanup
- **Action:** The user sees the final natural language answer.
- **Frontend Process:**
  - Streamlit displays the conversational answer directly in the chat interface.
  - It tucks the raw Generated SQL and the unformatted dictionary result inside a hidden, collapsible "View Details" dropdown for developers or curious users to inspect without cluttering the UI.
- **Cleanup:**
  - The user can optionally click the **"Clear Active Database"** button, which calls the `/clear_db` endpoint. This permanently deletes the hidden temporary `.db` file from the server's disk, resetting the session.
