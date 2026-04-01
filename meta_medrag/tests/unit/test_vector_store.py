"""
tests/unit/test_vector_store.py

Unit tests for Module 2 retrieval components.
Run with: pytest tests/unit/test_vector_store.py -v
"""

import numpy as np
import pytest


class TestMedicalVectorStore:

    @pytest.fixture
    def sample_reports(self):
        """Small set of fake medical reports for testing."""
        return [
            {"doc_id": f"doc_{i}", "text": f"FINDINGS: Patient shows finding number {i}. "
             f"The chest X-ray demonstrates typical pattern {i}.", "image_path": None}
            for i in range(20)
        ]

    @pytest.fixture
    def mock_text_encoder(self):
        """Deterministic mock encoder: hash text to fixed vector."""
        def encode(text: str) -> np.ndarray:
            np.random.seed(hash(text) % (2**31))
            v = np.random.randn(512).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            return v
        return encode

    def test_build_and_retrieve(self, sample_reports, mock_text_encoder, tmp_path):
        """Build a store and verify retrieval returns documents."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        store = MedicalVectorStore(
            domain="radiology",
            embedding_dim=512,
            store_dir=str(tmp_path),
        )
        store.build(sample_reports, text_encoder=mock_text_encoder)

        # Query with a random vector
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)

        docs = store.retrieve(query, max_k=3, similarity_ratio_threshold=0.0)

        assert len(docs) > 0, "Should retrieve at least one document"
        assert len(docs) <= 3, "Should not exceed max_k"
        assert docs[0].rank == 1, "First doc should have rank=1"
        assert all(d.domain == "radiology" for d in docs)

    def test_adaptive_k_filter(self, sample_reports, mock_text_encoder, tmp_path):
        """Higher similarity_ratio_threshold should return fewer documents."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        store = MedicalVectorStore("radiology", 512, str(tmp_path))
        store.build(sample_reports, mock_text_encoder)

        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)

        docs_loose  = store.retrieve(query, max_k=5, similarity_ratio_threshold=0.0)
        docs_strict = store.retrieve(query, max_k=5, similarity_ratio_threshold=0.99)

        assert len(docs_loose) >= len(docs_strict), \
            "Strict filter should return fewer or equal documents"

    def test_empty_store_returns_empty(self, tmp_path):
        """Querying an empty store should return empty list."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        store = MedicalVectorStore("radiology", 512, str(tmp_path))
        query = np.random.randn(512).astype(np.float32)

        docs = store.retrieve(query, max_k=5)
        assert docs == []

    def test_format_context(self, sample_reports, mock_text_encoder, tmp_path):
        """format_context should return non-empty string when docs exist."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        store = MedicalVectorStore("radiology", 512, str(tmp_path))
        store.build(sample_reports, mock_text_encoder)

        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        docs  = store.retrieve(query, max_k=3, similarity_ratio_threshold=0.0)

        context = store.format_context(docs, max_chars=1000)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Reference 1" in context

    def test_save_load_persistence(self, sample_reports, mock_text_encoder, tmp_path):
        """Store saved to disk should be loadable and give same results."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.module2_retrieval.vector_store import MedicalVectorStore

        store1 = MedicalVectorStore("radiology", 512, str(tmp_path))
        store1.build(sample_reports, mock_text_encoder)

        store2 = MedicalVectorStore("radiology", 512, str(tmp_path))  # loads from disk

        assert store2.index is not None
        assert store2.index.ntotal == len(sample_reports)
        assert len(store2.metadata) == len(sample_reports)


class TestDomainClassifier:

    def test_returns_valid_domain(self, tmp_path):
        """Classifier should always return a valid domain string."""
        try:
            import open_clip
        except ImportError:
            pytest.skip("open_clip not installed")

        from src.module2_retrieval.domain_classifier import DomainClassifier
        from PIL import Image

        clf = DomainClassifier(device="cpu")

        # Create a dummy white image
        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        domain, confidence, scores = clf.classify(img)

        valid_domains = {"radiology", "pathology", "ophthalmology"}
        assert domain in valid_domains, f"Invalid domain: {domain}"
        assert 0.0 <= confidence <= 1.0
        assert abs(sum(scores.values()) - 1.0) < 0.01, "Probabilities should sum to ~1"
