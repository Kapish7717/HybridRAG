from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import re
import torch.nn.functional as F
HALL_MODEL_NAME=os.getenv("HALL_MODEL_NAME","microsoft/deberta-large-mnli")

_tokenizer=None
_model =None
def load_hallucination_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            HALL_MODEL_NAME,
            # trust_remote_code=True,
            # use_fast=False  # IMPORTANT
        )

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            HALL_MODEL_NAME
        )
        _model.eval()

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
            probs = F.softmax(outputs.logits, dim=-1)
        # MNLI label mapping:
        # 0 = contradiction
        # 1 = neutral
        # 2 = entailment
        contradiction_score = probs[0][0].item()
        detailed.append((sentence, contradiction_score))

        if contradiction_score > threshold:
            hallucinated.append((sentence, contradiction_score))

    return hallucinated, detailed
