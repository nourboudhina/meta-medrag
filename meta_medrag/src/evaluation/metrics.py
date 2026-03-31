"""
src/evaluation/metrics.py

All evaluation metrics for Meta-MedRAG:
    VQA:            Accuracy, F1-score (macro)
    Report gen:     BLEU-4, ROUGE-L, METEOR
    System-level:   RAG trigger rate, avg MeCo score, avg latency
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    from sklearn.metrics import accuracy_score, f1_score
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import evaluate
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Install evaluation deps: pip install scikit-learn nltk rouge-score evaluate")


def compute_accuracy_f1(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """
    Compute token-level accuracy and macro F1 for VQA.

    For closed-ended VQA (yes/no/abnormal/normal answers),
    we compare the first meaningful token of prediction vs. reference.

    Args:
        predictions: list of model answers
        references:  list of ground-truth answers

    Returns:
        dict with "accuracy" and "f1"
    """
    def normalise(text: str) -> str:
        """Extract the key answer token."""
        text = text.lower().strip()
        # Handle common yes/no patterns
        if text.startswith(("yes", "abnormal", "present", "positive")):
            return "yes"
        if text.startswith(("no", "normal", "absent", "negative")):
            return "no"
        # Return first word for other cases
        return text.split()[0] if text else ""

    preds_norm = [normalise(p) for p in predictions]
    refs_norm  = [normalise(r) for r in references]

    acc = accuracy_score(refs_norm, preds_norm)
    f1  = f1_score(refs_norm, preds_norm, average="macro", zero_division=0)

    return {"accuracy": float(acc), "f1": float(f1)}


def compute_bleu4(
    predictions: List[str],
    references:  List[str],
) -> float:
    """
    Compute corpus-level BLEU-4.

    Args:
        predictions: list of generated reports
        references:  list of reference reports

    Returns:
        BLEU-4 score ∈ [0, 1]
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    tokenized_preds = [p.lower().split() for p in predictions]
    tokenized_refs  = [[r.lower().split()] for r in references]  # list of list of list

    smoothing = SmoothingFunction().method1
    score = corpus_bleu(
        tokenized_refs,
        tokenized_preds,
        weights=(0.25, 0.25, 0.25, 0.25),   # BLEU-4 uniform weights
        smoothing_function=smoothing,
    )
    return float(score)


def compute_rouge_l(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-L F1 scores.

    Args:
        predictions: list of generated reports
        references:  list of reference reports

    Returns:
        dict with "rouge_l_precision", "rouge_l_recall", "rouge_l_f1"
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    p_scores, r_scores, f_scores = [], [], []

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        p_scores.append(result["rougeL"].precision)
        r_scores.append(result["rougeL"].recall)
        f_scores.append(result["rougeL"].fmeasure)

    return {
        "rouge_l_precision": float(np.mean(p_scores)),
        "rouge_l_recall":    float(np.mean(r_scores)),
        "rouge_l_f1":        float(np.mean(f_scores)),
    }


def compute_meteor(
    predictions: List[str],
    references:  List[str],
) -> float:
    """
    Compute METEOR score using the HuggingFace evaluate library.

    Args:
        predictions: list of generated reports
        references:  list of reference reports

    Returns:
        METEOR score ∈ [0, 1]
    """
    try:
        meteor = evaluate.load("meteor")
        result = meteor.compute(predictions=predictions, references=references)
        return float(result["meteor"])
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
        return 0.0


def evaluate_vqa(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """Full VQA evaluation: accuracy + F1."""
    metrics = compute_accuracy_f1(predictions, references)
    logger.info(f"VQA — Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    return metrics


def evaluate_report_generation(
    predictions: List[str],
    references:  List[str],
) -> Dict[str, float]:
    """Full report generation evaluation: BLEU-4 + ROUGE-L + METEOR."""
    bleu4   = compute_bleu4(predictions, references)
    rouge   = compute_rouge_l(predictions, references)
    meteor  = compute_meteor(predictions, references)

    metrics = {
        "bleu4":          bleu4,
        "rouge_l_f1":     rouge["rouge_l_f1"],
        "rouge_l_recall": rouge["rouge_l_recall"],
        "meteor":         meteor,
    }

    logger.info(
        f"Report Gen — BLEU-4: {bleu4:.4f} | "
        f"ROUGE-L: {rouge['rouge_l_f1']:.4f} | "
        f"METEOR: {meteor:.4f}"
    )
    return metrics


def compute_system_metrics(pipeline_outputs: list) -> Dict[str, float]:
    """
    Compute system-level metrics from a list of PipelineOutput objects.
    These are unique to Meta-MedRAG — no baseline has these.

    Args:
        pipeline_outputs: list of PipelineOutput from pipeline.run_batch()

    Returns:
        dict with RAG statistics and latency numbers
    """
    from src.module1_metacognition.meco_probe import MeCoProbe

    total   = len(pipeline_outputs)
    direct  = sum(1 for o in pipeline_outputs if o.decision == MeCoProbe.DIRECT)
    soft    = sum(1 for o in pipeline_outputs if o.decision == MeCoProbe.SOFT_RAG)
    full    = sum(1 for o in pipeline_outputs if o.decision == MeCoProbe.FULL_RAG)

    avg_meco    = np.mean([o.meco_score for o in pipeline_outputs])
    avg_latency = np.mean([o.latency_ms for o in pipeline_outputs])
    avg_docs    = np.mean([o.n_docs_retrieved for o in pipeline_outputs])

    metrics = {
        "total_queries":       total,
        "direct_rate":         direct / total,
        "soft_rag_rate":       soft   / total,
        "full_rag_rate":       full   / total,
        "rag_trigger_rate":    (soft + full) / total,
        "avg_meco_score":      float(avg_meco),
        "avg_latency_ms":      float(avg_latency),
        "avg_docs_retrieved":  float(avg_docs),
    }

    logger.info(
        f"System — Direct: {direct/total:.1%} | "
        f"SoftRAG: {soft/total:.1%} | FullRAG: {full/total:.1%} | "
        f"Avg MeCo: {avg_meco:.3f} | Avg latency: {avg_latency:.0f}ms"
    )
    return metrics


def save_results(metrics: Dict, output_path: str, name: str = ""):
    """Save evaluation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"name": name, "metrics": metrics}, f, indent=2)
    logger.info(f"Results saved to {output_path}")
