"""
src/pipeline.py

Meta-MedRAG — full inference pipeline.

This is the single entry point for running the complete system.
It connects all three modules:

    Input (image + question)
        ↓
    [Module 1] Extract hidden states → MeCo score → dual threshold decision
        ├── DIRECT:   generate answer without retrieval
        ├── SOFT_RAG: retrieve 1 document → generate
        └── FULL_RAG: retrieve up to k documents → generate
                          ↓
                    [Module 2] Domain ID → FAISS retrieval → adaptive-k filter
                          ↓
                    [Module 3] DPO-aligned generation (image-first)
        ↓
    Output (answer + metadata)
"""

import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, List
from PIL import Image
from loguru import logger

from src.backbone.llava_med import LLaVAMedBackbone
from src.module1_metacognition.meco_probe import MeCoProbe
from src.module2_retrieval.domain_classifier import DomainClassifier
from src.module2_retrieval.vector_store import MultiDomainRetriever, RetrievedDocument


@dataclass
class PipelineOutput:
    """
    Structured output from one Meta-MedRAG inference call.
    Contains the answer plus full transparency metadata.
    Displayed in the Gradio interface.
    """
    # Core answer
    answer: str

    # Module 1 metadata
    meco_score: float
    decision:   str       # "direct" | "soft_rag" | "full_rag"

    # Module 2 metadata
    domain:         Optional[str]                  = None
    domain_confidence: Optional[float]             = None
    retrieved_docs: List[RetrievedDocument]        = field(default_factory=list)
    context_used:   Optional[str]                  = None

    # Timing
    latency_ms: float = 0.0

    @property
    def rag_triggered(self) -> bool:
        return self.decision in (MeCoProbe.SOFT_RAG, MeCoProbe.FULL_RAG)

    @property
    def n_docs_retrieved(self) -> int:
        return len(self.retrieved_docs)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"MeCo={self.meco_score:.3f} | decision={self.decision} | "
            f"domain={self.domain} | docs={self.n_docs_retrieved} | "
            f"latency={self.latency_ms:.0f}ms"
        )


class MetaMedRAGPipeline:
    """
    The complete Meta-MedRAG inference system.

    Initialise once, call run() for each query.

    Args:
        config_path: path to configs/config.yaml
        probe_path:  optional override for MeCo probe checkpoint path
        model_path:  optional override for LLaVA-Med model path
                     (use this to load DPO-fine-tuned checkpoint)
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        probe_path:  Optional[str] = None,
        model_path:  Optional[str] = None,
    ):
        logger.info("Initialising Meta-MedRAG pipeline...")
        t0 = time.time()

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Allow overrides
        if model_path:
            self.cfg["backbone"]["model_name"] = model_path

        # Load all components
        self._load_backbone()
        self._load_probe(probe_path)
        self._load_domain_classifier()
        self._load_retriever()

        elapsed = time.time() - t0
        logger.info(f"Pipeline ready in {elapsed:.1f}s")

    # ── Component loaders ─────────────────────────────────────────────

    def _load_backbone(self):
        logger.info("Loading backbone...")
        self.backbone = LLaVAMedBackbone(self.cfg["backbone"])

    def _load_probe(self, override_path: Optional[str] = None):
        logger.info("Loading MeCo probe...")
        mc_cfg = self.cfg["metacognition"]
        self.probe = MeCoProbe(
            n_components=mc_cfg["pca_components"],
            theta_low=mc_cfg["theta_low"],
            theta_high=mc_cfg["theta_high"],
        )
        probe_path = override_path or mc_cfg["probe_checkpoint"]
        if Path(probe_path).exists():
            self.probe.load(probe_path)
            logger.info(f"Probe loaded from {probe_path}")
        else:
            logger.warning(
                f"Probe checkpoint not found at {probe_path}. "
                "Run scripts/train_probe.py first."
            )

    def _load_domain_classifier(self):
        logger.info("Loading domain classifier...")
        self.domain_clf = DomainClassifier(
            model_name=self.cfg["retrieval"]["biomed_clip_model"],
            device=self.cfg["backbone"]["device"],
        )

    def _load_retriever(self):
        logger.info("Loading retriever...")
        self.retriever = MultiDomainRetriever(self.cfg)

    # ── Main inference ────────────────────────────────────────────────

    def run(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
    ) -> PipelineOutput:
        """
        Run the full Meta-MedRAG inference pipeline.

        Args:
            image:    medical image (file path or PIL Image)
            question: clinical question string

        Returns:
            PipelineOutput with answer + full transparency metadata
        """
        t_start = time.time()
        logger.info(f"Query: {question[:80]}...")

        # ── Step 1: Extract hidden states ────────────────────────────
        logger.debug("Step 1: Extracting hidden states...")
        hidden_states = self.backbone.get_hidden_states(
            image=image,
            question=question,
            layers=self.cfg["metacognition"]["probe_layers"],
        )

        # ── Step 2: Compute MeCo score & decide ──────────────────────
        logger.debug("Step 2: Computing MeCo score...")

        if self.probe._fitted:
            meco_score, decision = self.probe.score_and_decide(hidden_states)
        else:
            # Fallback if probe not trained: always do full RAG
            logger.warning("Probe not fitted — defaulting to FULL_RAG")
            meco_score = 0.8
            decision   = MeCoProbe.FULL_RAG

        logger.info(f"MeCo score: {meco_score:.3f} → {decision}")

        # ── Step 3: Direct path (no retrieval) ───────────────────────
        if decision == MeCoProbe.DIRECT:
            logger.debug("Step 3: Direct generation (no retrieval)")
            answer = self.backbone.generate(image, question)

            return PipelineOutput(
                answer=answer,
                meco_score=meco_score,
                decision=decision,
                latency_ms=(time.time() - t_start) * 1000,
            )

        # ── Step 4: Domain identification ────────────────────────────
        logger.debug("Step 4: Domain classification...")
        domain, domain_conf, _ = self.domain_clf.classify(image)
        logger.info(f"Domain: {domain} ({domain_conf:.2%})")

        # ── Step 5: Encode image for retrieval ────────────────────────
        logger.debug("Step 5: Encoding image for retrieval...")
        query_embedding = self.domain_clf.encode_image(image)

        # ── Step 6: Retrieve and filter contexts ──────────────────────
        logger.debug("Step 6: Retrieving documents...")
        docs, context = self.retriever.retrieve(
            query_embedding=query_embedding,
            domain=domain,
            decision=decision,
        )
        logger.info(f"Retrieved {len(docs)} documents")

        # ── Step 7: Generate with retrieved context ───────────────────
        logger.debug("Step 7: Generating answer with context...")
        answer = self.backbone.generate(
            image=image,
            question=question,
            retrieved_context=context if context else None,
        )

        latency = (time.time() - t_start) * 1000
        output = PipelineOutput(
            answer=answer,
            meco_score=meco_score,
            decision=decision,
            domain=domain,
            domain_confidence=domain_conf,
            retrieved_docs=docs,
            context_used=context,
            latency_ms=latency,
        )

        logger.info(f"Done. {output.summary()}")
        return output

    # ── Batch inference ───────────────────────────────────────────────

    def run_batch(
        self,
        items: List[dict],
        show_progress: bool = True,
    ) -> List[PipelineOutput]:
        """
        Run inference on a batch of items.

        Args:
            items: list of dicts with keys "image" and "question"
            show_progress: whether to show tqdm progress bar

        Returns:
            list of PipelineOutput
        """
        from tqdm import tqdm

        outputs = []
        iterator = tqdm(items, desc="Meta-MedRAG inference") if show_progress else items

        for item in iterator:
            output = self.run(item["image"], item["question"])
            outputs.append(output)

        # Summary stats
        rag_count  = sum(1 for o in outputs if o.rag_triggered)
        avg_lat    = sum(o.latency_ms for o in outputs) / len(outputs)
        avg_meco   = sum(o.meco_score for o in outputs) / len(outputs)

        logger.info(
            f"Batch complete: {len(outputs)} items | "
            f"RAG triggered: {rag_count}/{len(outputs)} ({rag_count/len(outputs):.1%}) | "
            f"avg MeCo: {avg_meco:.3f} | avg latency: {avg_lat:.0f}ms"
        )
        return outputs
