import os
import math
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def evaluate_rag(query: str, answer: str, contexts: list[str], ground_truth: str = None):
    """
    Runs Ragas evaluation with metrics: faithfulness, answer_relevancy, context_precision, context_recall.
    """
    try:
        print("\n📊 [Ragas] Starting Ragas Evaluation...")
        
        if not GROQ_API_KEY:
            return {"error": "GROQ_API_KEY not found! Cannot evaluate."}
            
        eval_llm = ChatGroq(model_name=RAGAS_LLM_MODEL, groq_api_key=GROQ_API_KEY)
        eval_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # Prepare data in the format Ragas expects
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        # Determine available metrics
        metrics = [faithfulness, answer_relevancy]
        
        if ground_truth:
            data["reference"] = [ground_truth]
            # Also keep ground_truth for older versions
            data["ground_truth"] = [ground_truth]
            metrics.extend([context_precision, context_recall])
        else:
            print("⚠️ ground_truth not provided. Skipping context_precision and context_recall.")

        dataset = Dataset.from_dict(data)

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        
        print("\n✅ [Ragas] Evaluation Results:")
        
        def safe_value(val):
            try:
                if val is None or math.isnan(float(val)):
                    return 0.0
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        metrics_output = {}
        if isinstance(result, dict):
            for m in metrics:
                metrics_output[m.name] = safe_value(result.get(m.name))
        else:
            df = result.to_pandas()
            for m in metrics:
                if m.name in df:
                    metrics_output[m.name] = safe_value(df[m.name].mean())
                else:
                    metrics_output[m.name] = 0.0
                    
        return metrics_output

    except Exception as e:
        print(f"\n❌ [Ragas] Evaluation failed: {str(e)}")
        return {"error": str(e)}


from textwrap import dedent

def generate_reference_answer(query: str, contexts: list[str]) -> str:
    """
    Uses a more powerful LLM to generate a 'Gold Standard' answer from retrieved context.
    This acts as the ground truth for RAGAS evaluation when no manual ground truth is provided.
    """
    try:
        print("\n🤖 [Reference] Generating automatic ground truth answer...")
        
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY not found!"
            
        eval_llm = ChatGroq(model_name=RAGAS_LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0)
        
        # Combine all contexts into a single block
        context_block = "\n\n---\n\n".join(contexts) if contexts else "No context available."
        
        prompt_template = dedent(f"""
            You are an expert evaluator and researcher.
            Your goal is to provide a "Gold Standard" answer based ONLY on the provided context.
            This answer will be used as the absolute ground truth to evaluate another AI's performance.

            RULES:
            1. Use ONLY the provided context. Do not use outside knowledge.
            2. Be extremely precise, complete, and concise.
            3. Do not include conversational filler like "Based on the context..." or "Here is the answer...".
            4. If the answer is not in the context, say "The provided context does not contain enough information to answer this question."

            CONTEXT:
            {context_block}

            USER QUESTION:
            {query}

            GOLD STANDARD GROUND TRUTH:
        """).strip()
        
        response = eval_llm.invoke(prompt_template)
        reference_answer = response.content.strip()
        
        print("✅ [Reference] Automatic ground truth generated successfully.")
        return reference_answer
        
    except Exception as e:
        print(f"❌ [Reference] Failed to generate reference answer: {str(e)}")
        return f"Error: Could not generate reference answer due to: {str(e)}"

    