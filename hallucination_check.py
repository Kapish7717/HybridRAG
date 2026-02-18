from transformers import AutoTokenizer, AutoModel
import os
import torch
import re
HALL_MODEL_NAME=os.getenv("HALL_MODEL_NAME","vectara/hallucination_evaluation_model")

_tokenizer=None
_model =None
def load_hallucination_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            HALL_MODEL_NAME,
            trust_remote_code=True,
            use_fast=False  # IMPORTANT
        )

    if _model is None:
        _model = AutoModel.from_pretrained(
            HALL_MODEL_NAME,
            trust_remote_code=True
        )

    return _tokenizer, _model


def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)


def check_hallucination(context:str,answer:str,threshold:float=0.5)->bool:
    """
    Returns:
        is_hallucinated (bool),
        score (float)
    """
    tokenizer,model = load_hallucination_model()
    sentences = split_into_sentences(answer)

    hallucinated = []
    detailed = []

    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue

        inputs = tokenizer(
            context,
            sentence,
            return_tensors="pt",
            truncation=True,
            # padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.squeeze().item()

        detailed.append((sentence, score))

        if score > threshold:
            hallucinated.append((sentence, score))

    return hallucinated, detailed