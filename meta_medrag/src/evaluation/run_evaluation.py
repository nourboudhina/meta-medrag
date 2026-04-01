"""
src/evaluation/run_evaluation.py

Full evaluation runner.
Evaluates Meta-MedRAG against baselines on the test split.

Usage:
    python -m src.evaluation.run_evaluation \
        --dataset mimic_cxr \
        --task vqa \
        --split test

Outputs:
    experiments/results/{dataset}_{task}_results.json
"""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from src.pipeline import MetaMedRAGPipeline
from src.evaluation.metrics import (
    evaluate_vqa, evaluate_report_generation,
    compute_system_metrics, save_results,
)


def load_test_data(cfg: dict, dataset: str, task: str, split: str = "test") -> list:
    """
    Load test items from a dataset split file.

    Expected split file format:
        [
            {
                "image": "path/to/image.jpg",
                "question": "Is there cardiomegaly?",
                "answer": "Yes",           // for VQA
                "report": "FINDINGS: ..."  // for report generation
            },
            ...
        ]
    """
    dataset_cfg  = cfg["datasets"][dataset]
    split_file   = dataset_cfg["split_file"]

    with open(split_file) as f:
        all_data = json.load(f)

    test_data = [item for item in all_data if item.get("split") == split]
    logger.info(f"Loaded {len(test_data)} {split} items from {dataset}")
    return test_data


def run_pipeline_on_dataset(
    pipeline: MetaMedRAGPipeline,
    test_data: list,
    task: str,
) -> tuple:
    """
    Run pipeline on all test items.

    Returns:
        (predictions, references, pipeline_outputs)
    """
    predictions = []
    references  = []
    outputs     = []

    ref_key = "answer" if task == "vqa" else "report"

    for item in tqdm(test_data, desc=f"Running {task} evaluation"):
        output = pipeline.run(item["image"], item["question"])
        predictions.append(output.answer)
        references.append(item.get(ref_key, ""))
        outputs.append(output)

    return predictions, references, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="mimic_cxr",
                        choices=["mimic_cxr", "iu_xray"])
    parser.add_argument("--task",       default="vqa",
                        choices=["vqa", "report_gen"])
    parser.add_argument("--split",      default="test")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--probe",      default=None)
    parser.add_argument("--model",      default=None,
                        help="Path to DPO fine-tuned model (optional)")
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load pipeline ────────────────────────────────────────────────
    pipeline = MetaMedRAGPipeline(
        config_path=args.config,
        probe_path=args.probe,
        model_path=args.model,
    )

    # ── Load data ────────────────────────────────────────────────────
    test_data = load_test_data(cfg, args.dataset, args.task, args.split)

    # ── Run inference ────────────────────────────────────────────────
    predictions, references, pipe_outputs = run_pipeline_on_dataset(
        pipeline, test_data, args.task
    )

    # ── Compute metrics ──────────────────────────────────────────────
    if args.task == "vqa":
        task_metrics = evaluate_vqa(predictions, references)
    else:
        task_metrics = evaluate_report_generation(predictions, references)

    system_metrics = compute_system_metrics(pipe_outputs)

    all_metrics = {**task_metrics, **system_metrics}

    # ── Save results ─────────────────────────────────────────────────
    out_name = f"{args.dataset}_{args.task}_{args.split}"
    out_path = f"{args.output_dir}/{out_name}_results.json"
    save_results(all_metrics, out_path, name=out_name)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"RESULTS: {args.dataset.upper()} — {args.task.upper()}")
    print("="*60)
    for k, v in all_metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("="*60)

    # Also save predictions for qualitative analysis
    pred_path = f"{args.output_dir}/{out_name}_predictions.json"
    with open(pred_path, "w") as f:
        json.dump([
            {
                "question":    test_data[i].get("question", ""),
                "reference":   references[i],
                "prediction":  predictions[i],
                "meco_score":  pipe_outputs[i].meco_score,
                "decision":    pipe_outputs[i].decision,
                "n_docs":      pipe_outputs[i].n_docs_retrieved,
            }
            for i in range(len(predictions))
        ], f, indent=2)
    logger.info(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
