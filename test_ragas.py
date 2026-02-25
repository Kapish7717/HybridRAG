import asyncio
from ragas_eval import run_ragas_evaluation

def test_callback(metrics):
    print("Callback received metrics:", metrics)

query = "What is the capital of France?"
answer = "The capital of France is Paris."
contexts = ["Paris is the capital and most populous city of France."]

print("Running test evaluation...")
result = run_ragas_evaluation(query, answer, contexts, callback=test_callback)
print("Result object:", result)
