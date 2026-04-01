"""
scripts/run_dpo_training.py

End-to-end script for Module 3 DPO fine-tuning.

Steps:
    1. Generate preference pairs (GPT-4o)
    2. Run DPO training
    3. Merge LoRA into base model

Usage:
    python scripts/run_dpo_training.py --step generate
    python scripts/run_dpo_training.py --step train
    python scripts/run_dpo_training.py --step merge
    python scripts/run_dpo_training.py --step all
"""

import argparse
import json
import yaml
import glob
import os
from pathlib import Path
from loguru import logger


def step_generate(cfg: dict):
    """Step 1: Generate preference pairs with GPT-4o."""
    from src.module3_alignment.preference_generator import PreferenceGenerator

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set OPENAI_API_KEY environment variable before generating preference pairs.\n"
            "  export OPENAI_API_KEY='sk-...'"
        )

    # Load radiology reports from MIMIC-CXR
    report_dir = cfg["datasets"]["mimic_cxr"]["report_dir"]
    image_dir  = cfg["datasets"]["mimic_cxr"]["image_dir"]

    reports = []
    for rf in list(Path(report_dir).glob("*.txt"))[:500]:  # limit to 500 for cost
        with open(rf) as f:
            text = f.read().strip()

        # Match to image
        img_candidates = list(Path(image_dir).glob(f"**/{rf.stem}*"))
        img_path = str(img_candidates[0]) if img_candidates else None

        reports.append({
            "doc_id":   rf.stem,
            "text":     text,
            "question": "Describe the key findings in this medical image.",
            "image_path": img_path,
        })

    logger.info(f"Loaded {len(reports)} reports for preference generation")

    generator = PreferenceGenerator(
        openai_api_key=api_key,
        model=cfg["openai"]["model"],
    )
    generator.generate_from_dataset(
        reports=reports,
        output_path=cfg["alignment"]["preference_data_path"],
        n_per_report=cfg["openai"]["n_pairs_per_report"],
    )
    logger.info("Preference pairs generated successfully")


def step_train(cfg: dict):
    """Step 2: Run DPO fine-tuning."""
    from src.module3_alignment.dpo_trainer import MedRAGDPOTrainer

    trainer = MedRAGDPOTrainer(cfg)
    output_dir = trainer.train()
    logger.info(f"DPO training complete. Checkpoint: {output_dir}")


def step_merge(cfg: dict):
    """Step 3: Merge LoRA weights into base model."""
    from src.module3_alignment.dpo_trainer import MedRAGDPOTrainer

    trainer = MedRAGDPOTrainer(cfg)
    lora_path   = cfg["alignment"]["output_dir"]
    merged_path = cfg["alignment"]["output_dir"] + "_merged"

    trainer.merge_and_save(
        lora_path=lora_path,
        output_path=merged_path,
    )
    logger.info(f"Merged model saved to {merged_path}")
    logger.info("Update configs/config.yaml backbone.model_name to point to the merged model")


def main():
    parser = argparse.ArgumentParser(description="Run DPO alignment training")
    parser.add_argument("--step", required=True,
                        choices=["generate", "train", "merge", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.step in ("generate", "all"):
        step_generate(cfg)
    if args.step in ("train", "all"):
        step_train(cfg)
    if args.step in ("merge", "all"):
        step_merge(cfg)


if __name__ == "__main__":
    main()
