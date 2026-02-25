import os
import math
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def run_ragas_evaluation(query: str, answer: str, contexts: list[str], callback=None):
    """
    Runs Ragas evaluation (Faithfulness and Answer Relevancy) in the background.
    """
    try:
        print("\n📊 [Ragas] Starting Ragas Evaluation in background...")
        # Initialize Groq and HF models for Ragas
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found! Cannot evaluate.")
            
        eval_llm = ChatGroq(model_name=RAGAS_LLM_MODEL, groq_api_key=GROQ_API_KEY)
        eval_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Prepare data in the format Ragas expects
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # We'll evaluate Faithfulness (is the answer grounded in context?) 
        # and Answer Relevancy (does it answer the user's question?)
        metrics = [faithfulness, answer_relevancy]
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=True # Temporarily enable to catch why Relevancy fails
        )
        
        print("\n✅ [Ragas] Evaluation Results:")
        metrics_output = None
        
        def safe_value(val):
            try:
                if val is None or math.isnan(float(val)):
                    return None
                return float(val)
            except (ValueError, TypeError):
                return None

        try:
            # For older Ragas versions
            if isinstance(result, dict):
                metrics_output = {
                    "faithfulness": safe_value(result.get('faithfulness')),
                    "answer_relevancy": safe_value(result.get('answer_relevancy'))
                }
                print(f"Faithfulness: {metrics_output['faithfulness']}")
                print(f"Answer Relevancy: {metrics_output['answer_relevancy']}")
            else:
                # For newer Ragas versions, try to convert to pandas and extract mean
                df = result.to_pandas()
                metrics_output = {
                    "faithfulness": safe_value(df['faithfulness'].mean() if 'faithfulness' in df else None),
                    "answer_relevancy": safe_value(df['answer_relevancy'].mean() if 'answer_relevancy' in df else None)
                }
                print(f"Faithfulness: {metrics_output['faithfulness']}")
                print(f"Answer Relevancy: {metrics_output['answer_relevancy']}")
        except Exception as e:
            print(f"Failed to parse results specifically, but evaluation finished: {str(e)}")
            metrics_output = {"error": "Failed to parse"}
            
        if callback and metrics_output:
            callback(metrics_output)
            
        return result
    except Exception as e:
        print(f"\n❌ [Ragas] Evaluation failed: {str(e)}")
