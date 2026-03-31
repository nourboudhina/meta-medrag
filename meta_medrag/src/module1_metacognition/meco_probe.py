"""
src/module1_metacognition/meco_probe.py

Meta-Cognition Probe — the heart of Module 1.

Pipeline:
    1. Load pre-extracted activations (from activation_extractor.py)
    2. Fit PCA to reduce dimensionality (e.g. 20480 → 20 components)
    3. Train a Logistic Regression probe on PCA features
    4. At inference: compute MeCo score = P(unknown | activations)
    5. Apply dual threshold to decide: direct / soft-RAG / full-RAG

This implements and EXTENDS the MeCo paper (Li et al., 2025):
    - Original MeCo: single threshold
    - Our innovation: DUAL THRESHOLD (theta_low, theta_high)
      creating a three-zone decision policy
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from loguru import logger


class MeCoProbe:
    """
    Meta-Cognition Probe with dual-threshold decision policy.

    Decision zones:
        score < theta_low  → DIRECT:   answer without retrieval
        score > theta_high → RAG:      full retrieval (k up to max_k)
        otherwise          → SOFT_RAG: minimal retrieval (k=1)

    Attributes:
        pca:        fitted PCA transformer
        scaler:     StandardScaler for feature normalisation
        probe:      fitted LogisticRegression classifier
        theta_low:  lower decision threshold
        theta_high: upper decision threshold
    """

    DIRECT   = "direct"
    SOFT_RAG = "soft_rag"
    FULL_RAG = "full_rag"

    def __init__(
        self,
        n_components: int = 20,
        theta_low:    float = 0.35,
        theta_high:   float = 0.65,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.theta_low    = theta_low
        self.theta_high   = theta_high
        self.random_state = random_state

        self.pca    = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.probe  = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # handles class imbalance
            C=1.0,
            random_state=random_state,
        )
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit PCA + scaler + probe on extracted activations.

        Args:
            X: activation matrix (N, feature_dim)
            y: binary labels — 0=known, 1=unknown/needs_retrieval

        Returns:
            dict with train/val accuracy, AUC
        """
        logger.info(f"Training MeCo probe on {X.shape[0]} samples, {X.shape[1]} features")

        # Train/val split (stratified to preserve class balance)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        # Step 1: Standardise
        X_tr_scaled  = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)

        # Step 2: PCA
        X_tr_pca  = self.pca.fit_transform(X_tr_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)

        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: {self.n_components} components explain {explained:.1%} of variance")

        # Step 3: Fit probe
        self.probe.fit(X_tr_pca, y_tr)
        self._fitted = True

        # Evaluate
        val_acc  = self.probe.score(X_val_pca, y_val)
        val_probs = self.probe.predict_proba(X_val_pca)[:, 1]
        val_auc  = roc_auc_score(y_val, val_probs)

        logger.info(f"Validation accuracy: {val_acc:.3f}")
        logger.info(f"Validation AUC-ROC:  {val_auc:.3f}")
        logger.info("\n" + classification_report(y_val, self.probe.predict(X_val_pca),
                                                  target_names=["known", "unknown"]))

        # Suggest optimal thresholds
        self._suggest_thresholds(val_probs, y_val)

        return {
            "val_accuracy": float(val_acc),
            "val_auc":      float(val_auc),
            "pca_variance": float(explained),
        }

    def _suggest_thresholds(self, probs: np.ndarray, y_true: np.ndarray):
        """
        Analyse the score distribution to suggest good threshold values.
        Prints percentile analysis to help choose theta_low and theta_high.
        """
        known_scores   = probs[y_true == 0]
        unknown_scores = probs[y_true == 1]

        logger.info("Score distribution analysis:")
        logger.info(f"  Known   (label=0): mean={known_scores.mean():.3f}, "
                    f"p90={np.percentile(known_scores, 90):.3f}")
        logger.info(f"  Unknown (label=1): mean={unknown_scores.mean():.3f}, "
                    f"p10={np.percentile(unknown_scores, 10):.3f}")
        logger.info(f"  Suggested theta_low  ≈ {np.percentile(known_scores, 85):.2f}")
        logger.info(f"  Suggested theta_high ≈ {np.percentile(unknown_scores, 20):.2f}")

    # ── Inference ─────────────────────────────────────────────────────

    def score(self, activations: Dict[int, np.ndarray]) -> float:
        """
        Compute MeCo score for a single query.

        Args:
            activations: dict {layer_idx: hidden_state_vector}
                         (output of backbone.get_hidden_states)

        Returns:
            MeCo score ∈ [0, 1]
            0 = model is confident it knows → no retrieval needed
            1 = model is uncertain → retrieval strongly recommended
        """
        if not self._fitted:
            raise RuntimeError("Probe not fitted. Call fit() or load() first.")

        # Concatenate layer activations in consistent order
        sorted_layers = sorted(activations.keys())
        concat = np.concatenate([activations[l] for l in sorted_layers], axis=0)

        # Apply same transforms as during training
        scaled = self.scaler.transform(concat.reshape(1, -1))
        pca_ft = self.pca.transform(scaled)

        # P(label=1) = P(unknown) = MeCo score
        meco_score = float(self.probe.predict_proba(pca_ft)[0, 1])
        return meco_score

    def decide(self, meco_score: float) -> str:
        """
        Apply dual-threshold decision policy.

        Args:
            meco_score: float ∈ [0, 1] from self.score()

        Returns:
            one of: MeCoProbe.DIRECT | MeCoProbe.SOFT_RAG | MeCoProbe.FULL_RAG
        """
        if meco_score < self.theta_low:
            return self.DIRECT
        elif meco_score > self.theta_high:
            return self.FULL_RAG
        else:
            return self.SOFT_RAG

    def score_and_decide(
        self, activations: Dict[int, np.ndarray]
    ) -> Tuple[float, str]:
        """Convenience: compute score and decision in one call."""
        s = self.score(activations)
        d = self.decide(s)
        return s, d

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted probe to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pca":          self.pca,
            "scaler":       self.scaler,
            "probe":        self.probe,
            "theta_low":    self.theta_low,
            "theta_high":   self.theta_high,
            "n_components": self.n_components,
            "_fitted":      self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"MeCo probe saved to {path}")

    def load(self, path: str) -> "MeCoProbe":
        """Load probe from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.pca          = state["pca"]
        self.scaler       = state["scaler"]
        self.probe        = state["probe"]
        self.theta_low    = state["theta_low"]
        self.theta_high   = state["theta_high"]
        self.n_components = state["n_components"]
        self._fitted      = state["_fitted"]
        logger.info(f"MeCo probe loaded from {path}")
        return self


# ── Script: train probe ───────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, pickle

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mc_cfg = cfg["metacognition"]

    # Load pre-extracted activations
    with open("data/processed/activations_train.pkl", "rb") as f:
        data = pickle.load(f)

    X, y = data["X"], data["y"]

    # Train probe
    probe = MeCoProbe(
        n_components=mc_cfg["pca_components"],
        theta_low=mc_cfg["theta_low"],
        theta_high=mc_cfg["theta_high"],
    )
    results = probe.fit(X, y)

    print(f"\nFinal results: {results}")

    # Save
    probe.save(mc_cfg["probe_checkpoint"])
