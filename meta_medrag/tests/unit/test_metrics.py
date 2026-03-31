"""
tests/unit/test_metrics.py

Unit tests for evaluation metrics.
Run with: pytest tests/unit/test_metrics.py -v
"""

import pytest


class TestVQAMetrics:

    def test_perfect_accuracy(self):
        """When predictions == references, accuracy should be 1.0."""
        from src.evaluation.metrics import compute_accuracy_f1

        preds = ["yes", "no", "yes", "no"]
        refs  = ["yes", "no", "yes", "no"]
        m = compute_accuracy_f1(preds, refs)

        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_zero_accuracy(self):
        """When all predictions are wrong, accuracy should be 0.0."""
        from src.evaluation.metrics import compute_accuracy_f1

        preds = ["yes", "yes", "yes"]
        refs  = ["no",  "no",  "no"]
        m = compute_accuracy_f1(preds, refs)

        assert m["accuracy"] == 0.0

    def test_normalisation(self):
        """Predictions like 'Yes, there is...' should normalise to 'yes'."""
        from src.evaluation.metrics import compute_accuracy_f1

        preds = ["Yes, there is cardiomegaly.", "No findings detected."]
        refs  = ["yes", "no"]
        m = compute_accuracy_f1(preds, refs)

        assert m["accuracy"] == 1.0

    def test_mixed_predictions(self):
        """Partial matches should give accuracy between 0 and 1."""
        from src.evaluation.metrics import compute_accuracy_f1

        preds = ["yes", "yes", "no", "no"]
        refs  = ["yes", "no",  "no", "yes"]
        m = compute_accuracy_f1(preds, refs)

        assert 0.0 < m["accuracy"] < 1.0


class TestReportGenerationMetrics:

    def test_bleu4_identical(self):
        """Identical prediction and reference should give BLEU-4 = 1.0."""
        from src.evaluation.metrics import compute_bleu4

        text  = "The chest X-ray shows clear lung fields with no evidence of consolidation."
        score = compute_bleu4([text], [text])

        assert score > 0.9, f"BLEU-4 for identical text should be near 1.0, got {score}"

    def test_bleu4_completely_different(self):
        """Completely different texts should give very low BLEU-4."""
        from src.evaluation.metrics import compute_bleu4

        pred = "The patient is healthy."
        ref  = "Severe bilateral pneumonia with consolidation in lower lobes."
        score = compute_bleu4([pred], [ref])

        assert score < 0.1, f"BLEU-4 for unrelated text should be low, got {score}"

    def test_rouge_l_range(self):
        """ROUGE-L F1 should be in [0, 1]."""
        from src.evaluation.metrics import compute_rouge_l

        preds = ["cardiomegaly is present", "no pleural effusion"]
        refs  = ["cardiomegaly noted",      "bilateral effusion observed"]
        m = compute_rouge_l(preds, refs)

        assert 0.0 <= m["rouge_l_f1"]     <= 1.0
        assert 0.0 <= m["rouge_l_recall"] <= 1.0

    def test_empty_prediction(self):
        """Empty predictions should not crash."""
        from src.evaluation.metrics import compute_accuracy_f1, compute_bleu4

        m = compute_accuracy_f1([""], ["yes"])
        assert "accuracy" in m

        score = compute_bleu4([""], ["some reference text here"])
        assert isinstance(score, float)


class TestSystemMetrics:

    def test_rag_rates_sum_to_one(self):
        """Direct + soft_rag + full_rag rates should sum to 1."""
        from src.evaluation.metrics import compute_system_metrics

        # Mock PipelineOutput objects
        class MockOutput:
            def __init__(self, decision, meco_score):
                self.decision       = decision
                self.meco_score     = meco_score
                self.latency_ms     = 100.0
                self.n_docs_retrieved = 0 if decision == "direct" else 2

        outputs = [
            MockOutput("direct",   0.2),
            MockOutput("soft_rag", 0.5),
            MockOutput("full_rag", 0.8),
            MockOutput("direct",   0.1),
        ]

        m = compute_system_metrics(outputs)

        total = m["direct_rate"] + m["soft_rag_rate"] + m["full_rag_rate"]
        assert abs(total - 1.0) < 1e-6, f"Rates should sum to 1.0, got {total}"

        assert m["rag_trigger_rate"] == m["soft_rag_rate"] + m["full_rag_rate"]
        assert m["total_queries"] == 4
