"""
scripts/train_probe.py

Step-by-step script to train the Module 1 meta-cognition probe.

Run in order:
    1. python scripts/train_probe.py --step build_pairs
    2. python scripts/train_probe.py --step extract
    3. python scripts/train_probe.py --step train
    4. python scripts/train_probe.py --step evaluate
"""

import argparse
import yaml
import pickle
import json
import glob
from pathlib import Path
from loguru import logger


def step_build_pairs(cfg: dict, args):
    """Step 1: Build contrastive dataset from all available datasets."""
    from src.module1_metacognition.contrastive_dataset import ContrastiveDatasetBuilder
    import os

    logger.info("Building contrastive dataset from all datasets...")

    builder = ContrastiveDatasetBuilder(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        seed=42,
    )
    output_path = cfg["metacognition"]["contrastive_data_path"]
    builder.build_from_all_datasets(
        output_path=output_path,
        n_per_source=150,
    )
    logger.info("Done: contrastive_pairs.json created")

def step_extract(cfg: dict, args):
    """Step 2: Extract activations from LLaVA-Med."""
    from src.backbone.llava_med import LLaVAMedBackbone
    from src.module1_metacognition.activation_extractor import ActivationExtractor

    backbone  = LLaVAMedBackbone(cfg["backbone"])
    extractor = ActivationExtractor(
        backbone=backbone,
        layers=cfg["metacognition"]["probe_layers"],
        output_dir="data/processed",
    )
    extractor.extract_and_save(
        data_path=cfg["metacognition"]["contrastive_data_path"],
        output_prefix="activations_train",
    )
    logger.info("Done: activations_train.pkl created")


def step_train(cfg: dict, args):
    """Step 3: Train PCA + linear probe."""
    from src.module1_metacognition.meco_probe import MeCoProbe

    mc_cfg = cfg["metacognition"]

    with open("data/processed/activations_train.pkl", "rb") as f:
        data = pickle.load(f)

    X, y = data["X"], data["y"]
    logger.info(f"Training on X={X.shape}, y={y.shape}")

    probe = MeCoProbe(
        n_components=mc_cfg["pca_components"],
        theta_low=mc_cfg["theta_low"],
        theta_high=mc_cfg["theta_high"],
    )
    results = probe.fit(X, y)

    probe.save(mc_cfg["probe_checkpoint"])
    logger.info(f"Probe saved. Results: {results}")


def step_evaluate(cfg: dict, args):
    """Step 4: Evaluate probe on a hold-out set."""
    from src.module1_metacognition.meco_probe import MeCoProbe
    import numpy as np
    from sklearn.model_selection import cross_val_score

    mc_cfg = cfg["metacognition"]

    with open("data/processed/activations_train.pkl", "rb") as f:
        data = pickle.load(f)

    X, y = data["X"], data["y"]

    probe = MeCoProbe()
    probe.load(mc_cfg["probe_checkpoint"])

    # 5-fold cross-validation
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=mc_cfg["pca_components"])),
        ("probe",  LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    logger.info(f"5-fold CV AUC-ROC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Score distribution
    scaled = probe.scaler.transform(X)
    pca_ft = probe.pca.transform(scaled)
    probs  = probe.probe.predict_proba(pca_ft)[:, 1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[y == 0], bins=30, alpha=0.6, label="Known (label=0)", color="#1D9E75")
    ax.hist(probs[y == 1], bins=30, alpha=0.6, label="Unknown (label=1)", color="#993C1D")
    ax.axvline(mc_cfg["theta_low"],  color="blue",   linestyle="--", label=f"θ_low={mc_cfg['theta_low']}")
    ax.axvline(mc_cfg["theta_high"], color="orange", linestyle="--", label=f"θ_high={mc_cfg['theta_high']}")
    ax.set_xlabel("MeCo score")
    ax.set_ylabel("Count")
    ax.set_title("MeCo score distribution by class")
    ax.legend()
    plt.tight_layout()
    plt.savefig("experiments/results/meco_score_distribution.png", dpi=150)
    logger.info("Plot saved to experiments/results/meco_score_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Train Meta-MedRAG Module 1 probe")
    parser.add_argument("--step", required=True,
                        choices=["build_pairs", "extract", "train", "evaluate", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.step in ("build_pairs", "all"):
        step_build_pairs(cfg, args)
    if args.step in ("extract", "all"):
        step_extract(cfg, args)
    if args.step in ("train", "all"):
        step_train(cfg, args)
    if args.step in ("evaluate", "all"):
        step_evaluate(cfg, args)


if __name__ == "__main__":
    main()
