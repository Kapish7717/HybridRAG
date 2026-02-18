from rag import generate_answer
from ingestion import ingest_pdf
from rag import generate_answer



query = input("Write the query:")

answer = generate_answer(query)
print(answer)


