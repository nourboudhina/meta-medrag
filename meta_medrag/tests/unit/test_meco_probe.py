"""
tests/unit/test_meco_probe.py

Unit tests for the MeCo probe (Module 1).
Run with: pytest tests/unit/test_meco_probe.py -v
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """
    Generate synthetic activation data for testing.
    Known (label=0): activations centred near 0
    Unknown (label=1): activations centred near 1
    """
    np.random.seed(42)
    n_samples   = 200
    feature_dim = 5 * 4096  # 5 layers × 4096 hidden dim

    X_known   = np.random.randn(n_samples // 2, feature_dim) * 0.5
    X_unknown = np.random.randn(n_samples // 2, feature_dim) * 0.5 + 1.0

    X = np.vstack([X_known, X_unknown])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def trained_probe(synthetic_data):
    """Return a fitted MeCoProbe instance."""
    from src.module1_metacognition.meco_probe import MeCoProbe

    X, y = synthetic_data
    probe = MeCoProbe(n_components=10, theta_low=0.35, theta_high=0.65)
    probe.fit(X, y)
    return probe


# ── Tests ─────────────────────────────────────────────────────────────

class TestMeCoProbe:

    def test_fit_returns_metrics(self, synthetic_data):
        """probe.fit() should return accuracy and AUC above 0.6."""
        from src.module1_metacognition.meco_probe import MeCoProbe

        X, y = synthetic_data
        probe = MeCoProbe(n_components=10)
        results = probe.fit(X, y)

        assert "val_accuracy" in results
        assert "val_auc"      in results
        assert results["val_accuracy"] > 0.6, "Probe accuracy too low on synthetic data"
        assert results["val_auc"]      > 0.6, "Probe AUC too low on synthetic data"

    def test_score_range(self, trained_probe, synthetic_data):
        """MeCo score should always be in [0, 1]."""
        X, y = synthetic_data

        # Create fake activations dict matching the layer structure
        layers = [-2, -5, -8, -11, -15]
        chunk  = X.shape[1] // len(layers)

        for i in range(10):
            activations = {
                l: X[i, j*chunk:(j+1)*chunk]
                for j, l in enumerate(layers)
            }
            score = trained_probe.score(activations)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_decision_thresholds(self, trained_probe):
        """Dual-threshold decision logic should produce correct zones."""
        from src.module1_metacognition.meco_probe import MeCoProbe

        # Force specific scores to test decision boundaries
        assert trained_probe.decide(0.1)  == MeCoProbe.DIRECT
        assert trained_probe.decide(0.34) == MeCoProbe.DIRECT
        assert trained_probe.decide(0.35) == MeCoProbe.SOFT_RAG
        assert trained_probe.decide(0.50) == MeCoProbe.SOFT_RAG
        assert trained_probe.decide(0.65) == MeCoProbe.SOFT_RAG
        assert trained_probe.decide(0.66) == MeCoProbe.FULL_RAG
        assert trained_probe.decide(0.99) == MeCoProbe.FULL_RAG

    def test_save_load_roundtrip(self, trained_probe, synthetic_data, tmp_path):
        """Saving and loading probe should preserve scores."""
        X, y = synthetic_data
        layers = [-2, -5, -8, -11, -15]
        chunk  = X.shape[1] // len(layers)

        activations = {
            l: X[0, j*chunk:(j+1)*chunk]
            for j, l in enumerate(layers)
        }

        score_before = trained_probe.score(activations)

        # Save and reload
        save_path = str(tmp_path / "test_probe.pkl")
        trained_probe.save(save_path)

        from src.module1_metacognition.meco_probe import MeCoProbe
        loaded = MeCoProbe()
        loaded.load(save_path)

        score_after = loaded.score(activations)

        assert abs(score_before - score_after) < 1e-6, \
            f"Score changed after save/load: {score_before} → {score_after}"

    def test_unfitted_probe_raises(self):
        """Calling score() before fit() should raise RuntimeError."""
        from src.module1_metacognition.meco_probe import MeCoProbe

        probe = MeCoProbe()
        with pytest.raises(RuntimeError, match="not fitted"):
            probe.score({-2: np.random.randn(4096)})

    def test_custom_thresholds(self):
        """Custom theta_low and theta_high should be respected."""
        from src.module1_metacognition.meco_probe import MeCoProbe

        probe = MeCoProbe(theta_low=0.2, theta_high=0.8)
        assert probe.decide(0.19) == MeCoProbe.DIRECT
        assert probe.decide(0.50) == MeCoProbe.SOFT_RAG
        assert probe.decide(0.81) == MeCoProbe.FULL_RAG
