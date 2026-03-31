"""
tests/integration/test_pipeline_integration.py

Integration test: runs the full pipeline end-to-end with mocked components.
Tests that all modules connect correctly without requiring real GPU or data.

Run with: pytest tests/integration/ -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


@pytest.fixture
def dummy_image():
    """A blank 224x224 RGB image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def mock_cfg():
    """Minimal config dict for testing."""
    return {
        "backbone": {
            "model_name": "mock/llava-med",
            "device": "cpu",
            "torch_dtype": "float16",
            "max_new_tokens": 64,
            "temperature": 0.1,
            "do_sample": False,
        },
        "metacognition": {
            "probe_layers": [-2, -5],
            "pca_components": 5,
            "theta_low": 0.35,
            "theta_high": 0.65,
            "probe_checkpoint": "/tmp/test_probe.pkl",
            "contrastive_data_path": "/tmp/contrastive.json",
        },
        "retrieval": {
            "biomed_clip_model": "mock/biomed-clip",
            "vector_store_dir": "/tmp/vector_stores",
            "similarity_ratio_threshold": 0.85,
            "max_k": 3,
            "min_k": 1,
            "embedding_dim": 64,
            "domains": ["radiology"],
        },
        "alignment": {
            "base_model": "mock/llava-med",
            "output_dir": "/tmp/dpo_output",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj"],
            "dpo_beta": 0.1,
            "learning_rate": 5e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.0,
            "max_length": 256,
            "fp16": False,
            "preference_data_path": "/tmp/prefs.json",
        },
    }


class TestMeCoProbeIntegration:

    def test_probe_decision_flow(self, mock_cfg):
        """MeCoProbe should make decisions on synthetic activations."""
        from src.module1_metacognition.meco_probe import MeCoProbe

        # Generate synthetic training data
        np.random.seed(0)
        n, d = 100, 5 * 512   # 5 layers × 512 dim (small for tests)

        X_known   = np.random.randn(n//2, d) * 0.3
        X_unknown = np.random.randn(n//2, d) * 0.3 + 2.0
        X = np.vstack([X_known, X_unknown])
        y = np.array([0]*(n//2) + [1]*(n//2))

        probe = MeCoProbe(
            n_components=mock_cfg["metacognition"]["pca_components"],
            theta_low=mock_cfg["metacognition"]["theta_low"],
            theta_high=mock_cfg["metacognition"]["theta_high"],
        )
        results = probe.fit(X, y)

        # Should fit without error
        assert "val_accuracy" in results
        assert probe._fitted

        # Score and decide on a "known" sample
        layers   = [-2, -5]
        chunk    = d // len(layers)
        known_acts = {l: X_known[0, i*chunk:(i+1)*chunk] for i, l in enumerate(layers)}
        unknown_acts = {l: X_unknown[0, i*chunk:(i+1)*chunk] for i, l in enumerate(layers)}

        score_known   = probe.score(known_acts)
        score_unknown = probe.score(unknown_acts)

        # Known samples should have lower MeCo scores
        assert score_known < score_unknown, \
            f"Known score ({score_known:.3f}) should be < unknown score ({score_unknown:.3f})"


class TestPipelineOutputStructure:

    def test_pipeline_output_properties(self):
        """PipelineOutput should correctly compute derived properties."""
        from src.pipeline import PipelineOutput
        from src.module2_retrieval.vector_store import RetrievedDocument

        doc = RetrievedDocument(
            doc_id="test", text="...", image_path=None,
            domain="radiology", score=0.9, rank=1
        )

        # Direct path output
        out_direct = PipelineOutput(
            answer="No cardiomegaly",
            meco_score=0.2,
            decision="direct",
        )
        assert not out_direct.rag_triggered
        assert out_direct.n_docs_retrieved == 0

        # RAG path output
        out_rag = PipelineOutput(
            answer="Mild cardiomegaly noted",
            meco_score=0.8,
            decision="full_rag",
            domain="radiology",
            domain_confidence=0.92,
            retrieved_docs=[doc],
            context_used="Reference: ...",
            latency_ms=450.0,
        )
        assert out_rag.rag_triggered
        assert out_rag.n_docs_retrieved == 1
        assert "full_rag" in out_rag.summary()
        assert "0.800" in out_rag.summary()


class TestVectorStoreIntegration:

    def test_build_retrieve_cycle(self, tmp_path):
        """Full build → retrieve cycle without GPU."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        # Deterministic encoder
        def enc(text: str) -> np.ndarray:
            np.random.seed(abs(hash(text)) % (2**31))
            v = np.random.randn(64).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)

        reports = [
            {"doc_id": f"r{i}", "text": f"Radiology report {i} showing finding {i}",
             "image_path": None}
            for i in range(10)
        ]

        store = MedicalVectorStore("radiology", embedding_dim=64, store_dir=str(tmp_path))
        store.build(reports, text_encoder=enc)

        query = np.random.randn(64).astype(np.float32)
        query = query / np.linalg.norm(query)

        docs = store.retrieve(query, max_k=3, similarity_ratio_threshold=0.0)
        assert len(docs) == 3
        assert all(isinstance(d.score, float) for d in docs)
        assert docs[0].score >= docs[-1].score  # sorted by score descending
