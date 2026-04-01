"""
src/module1_metacognition/activation_extractor.py

Extracts and saves hidden-state activations from LLaVA-Med
for a dataset of (image, question, label) triples.

Run this script ONCE to build the activation dataset used
to train the meta-cognition probe.

Labels:
    0 = model knows the answer (no retrieval needed)
    1 = model does NOT know the answer (retrieval needed)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from loguru import logger

from src.backbone.llava_med import LLaVAMedBackbone


class ActivationExtractor:
    """
    Orchestrates hidden-state extraction for the probe training dataset.

    Workflow:
        1. Load a dataset of contrastive pairs (known vs. unknown questions)
        2. For each sample, run LLaVA-Med forward pass
        3. Collect activations at specified transformer layers
        4. Save as numpy arrays for fast loading during probe training

    Expected input format (contrastive_pairs.json):
        [
            {
                "image": "data/raw/mimic_cxr/images/p10/s12345/img.jpg",
                "question": "Is there evidence of cardiomegaly?",
                "label": 0,      // 0=known, 1=unknown
                "domain": "radiology"
            },
            ...
        ]
    """

    def __init__(
        self,
        backbone: LLaVAMedBackbone,
        layers: List[int] = [-2, -5, -8, -11, -15],
        output_dir: str = "data/processed",
    ):
        self.backbone   = backbone
        self.layers     = layers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_contrastive_pairs(self, json_path: str) -> List[Dict]:
        """Load contrastive pair dataset from JSON."""
        with open(json_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} contrastive pairs from {json_path}")

        # Validate
        required_keys = {"image", "question", "label"}
        for i, item in enumerate(data):
            missing = required_keys - set(item.keys())
            if missing:
                raise ValueError(f"Item {i} missing keys: {missing}")

        return data

    def extract_and_save(
        self,
        data_path: str,
        output_prefix: str = "activations",
        batch_size: int = 4,
    ) -> str:
        """
        Extract activations for all items in the dataset and save to disk.

        Args:
            data_path:     path to contrastive_pairs.json
            output_prefix: prefix for output files
            batch_size:    items per GPU batch

        Returns:
            Path to the saved activation file (.pkl)
        """
        data = self.load_contrastive_pairs(data_path)

        all_features = []   # shape: (N, n_layers * hidden_dim)
        all_labels   = []   # shape: (N,)

        logger.info(f"Extracting activations for {len(data)} samples...")

        for item in tqdm(data, desc="Extracting activations"):
            try:
                states = self.backbone.get_hidden_states(
                    image=item["image"],
                    question=item["question"],
                    layers=self.layers,
                )

                # Concatenate activations from all layers into one flat vector
                # e.g. 5 layers × 4096 dim = 20480-dim vector
                concat = np.concatenate(
                    [states[l] for l in self.layers], axis=0
                )
                all_features.append(concat)
                all_labels.append(item["label"])

            except Exception as e:
                logger.warning(f"Skipping item (error: {e}): {item.get('question', '')[:60]}")
                continue

        X = np.stack(all_features, axis=0)     # (N, feature_dim)
        y = np.array(all_labels, dtype=np.int32)  # (N,)

        logger.info(f"Extracted activations: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: known={int((y==0).sum())}, unknown={int((y==1).sum())}")

        # Save
        out_path = self.output_dir / f"{output_prefix}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"X": X, "y": y, "layers": self.layers}, f)

        logger.info(f"Saved activations to {out_path}")
        return str(out_path)

    def load_activations(self, pkl_path: str) -> Dict:
        """Load previously extracted activations."""
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded activations: X={data['X'].shape}, y={data['y'].shape}")
        return data


# ── Script entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

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
