from functools import lru_cache

import numpy as np
import torch
from torchmetrics.functional.text import bert_score, bleu_score
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.util import get_device


@lru_cache(1)
def _get_by_name(
    model_name: str,
) -> tuple[RobertaForSequenceClassification, RobertaTokenizer]:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    model.to(get_device())
    return model, tokenizer


def classify_predictions(
    predictions,
    soft=False,
    model_name: str = "SkolkovoInstitute/roberta_toxicity_classifier_v1",
    batch_size: int = 64,
    threshold: float = 0.8,
) -> float:
    """
    Evaluates predictions using Roberta toxicity classifier.
    Returns mean non-toxicity score (1 - toxicity by model), bigger is better.

    Source: https://github.com/s-nlp/detox/blob/main/emnlp2021/metric/metric.py#L27
    """
    results = []

    model, tokenizer = _get_by_name(model_name)

    for i in range(0, len(predictions), batch_size):
        batch = tokenizer(
            predictions[i : i + batch_size], return_tensors="pt", padding=True
        ).to(get_device())
        with torch.inference_mode():
            logits: torch.Tensor = model(**batch).logits
        if soft:
            result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        else:
            result = (logits[:, 1] > threshold).cpu().numpy()
        results.extend([1 - item for item in result])
    return np.mean(results)


def mean_bert(
    predictions: list[str],
    source: list[str],
    return_stat: str = "f1",
    batch_size: int = 64,
) -> float:
    bert_stats = bert_score(
        predictions,
        source,
        device=get_device(),
        model_name_or_path="bert-base-uncased",
        batch_size=batch_size,
    )
    return bert_stats[return_stat].mean().item()


def mean_bleu(predictions: list[str], target: list[str]) -> float:
    return bleu_score(predictions, target).mean().item()


def calculate_all(
    source: list[str], predictions: list[str], target: list[str], batch_size: int = 64
) -> tuple[float, float, float]:
    """Computes non-toxicity score, BERT similarity and BLEU score"""
    non_tox_score = classify_predictions(predictions, batch_size=batch_size)
    bert = mean_bert(predictions, source, batch_size=batch_size)
    bleu = mean_bleu(predictions, target)
    return non_tox_score, bert, bleu
