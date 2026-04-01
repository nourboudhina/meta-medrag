"""
src/module2_retrieval/vector_store.py

FAISS-based vector store for medical report retrieval.

For each medical domain, we maintain a separate FAISS index:
    - data/vector_stores/radiology.index + radiology_meta.json
    - data/vector_stores/pathology.index + pathology_meta.json
    - data/vector_stores/ophthalmology.index + ophthalmology_meta.json

Each document in the index is a medical report embedding,
encoded with BiomedCLIP text encoder.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed — install with: pip install faiss-gpu")


@dataclass
class RetrievedDocument:
    """One retrieved medical document with its similarity score."""
    doc_id:     str
    text:       str       # full report text
    image_path: Optional[str]
    domain:     str
    score:      float     # cosine similarity score ∈ [-1, 1]
    rank:       int       # 1 = most similar


class MedicalVectorStore:
    """
    FAISS vector store for a single medical domain.

    Builds:
        1. Embed all reports with BiomedCLIP text encoder
        2. Add embeddings to a FAISS IndexFlatIP (inner product = cosine similarity)
        3. Save index + metadata to disk

    Retrieves:
        1. Embed query with BiomedCLIP image encoder
        2. Search FAISS index for top-k nearest neighbours
        3. Apply adaptive-k filter (similarity ratio)
        4. Return filtered RetrievedDocument list
    """

    def __init__(
        self,
        domain: str,
        embedding_dim: int = 512,
        store_dir: str = "data/vector_stores",
    ):
        self.domain        = domain
        self.embedding_dim = embedding_dim
        self.store_dir     = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_dir / f"{domain}.index"
        self.meta_path  = self.store_dir / f"{domain}_meta.json"

        self.index    = None  # FAISS index
        self.metadata = []    # list of dicts: {doc_id, text, image_path}

        # Try to load existing index
        if self.index_path.exists():
            self._load()

    # ── Building the index ────────────────────────────────────────────

    def build(
        self,
        reports: List[Dict],
        text_encoder,           # BiomedCLIP text encoder callable
        batch_size: int = 64,
    ):
        """
        Build FAISS index from a list of medical reports.

        Args:
            reports: list of dicts with keys "doc_id", "text", "image_path"
            text_encoder: callable(text: str) → np.ndarray of shape (embedding_dim,)
            batch_size: reports to encode per batch

        Example report dict:
            {"doc_id": "mimic_s12345", "text": "FINDINGS: ...", "image_path": "..."}
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Run: pip install faiss-gpu")

        logger.info(f"Building FAISS index for domain={self.domain}, n={len(reports)}")

        # FAISS IndexFlatIP: inner product search (equivalent to cosine if normalised)
        self.index    = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []

        all_embeddings = []
        for i in range(0, len(reports), batch_size):
            batch = reports[i : i + batch_size]
            for doc in batch:
                emb = text_encoder(doc["text"])          # (embedding_dim,)
                emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2 normalise
                all_embeddings.append(emb)
                self.metadata.append({
                    "doc_id":     doc["doc_id"],
                    "text":       doc["text"],
                    "image_path": doc.get("image_path"),
                    "domain":     self.domain,
                })
            logger.debug(f"Encoded {min(i+batch_size, len(reports))}/{len(reports)} reports")

        embeddings_matrix = np.stack(all_embeddings, axis=0).astype(np.float32)
        self.index.add(embeddings_matrix)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors")
        self._save()

    # ── Retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        max_k: int = 5,
        similarity_ratio_threshold: float = 0.85,
    ) -> List[RetrievedDocument]:
        """
        Retrieve and filter top-k documents for a query embedding.

        Adaptive-k filtering (from RULE, Xia et al. 2024):
            - Retrieve max_k candidates
            - Compute similarity ratio: score_i / score_1
            - Drop documents where ratio < similarity_ratio_threshold
            - This automatically adjusts k to the query's retrievability

        Args:
            query_embedding:           L2-normalised query vector (embedding_dim,)
            max_k:                     initial number of candidates to retrieve
            similarity_ratio_threshold: filter threshold for adaptive k

        Returns:
            List of RetrievedDocument, filtered and ranked by similarity
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning(f"Empty index for domain={self.domain}")
            return []

        # Normalise query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)

        # FAISS search
        k       = min(max_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        scores  = scores[0]   # shape (k,)
        indices = indices[0]  # shape (k,)

        if len(scores) == 0 or scores[0] <= 0:
            return []

        # Adaptive-k filter
        top_score = scores[0]
        docs      = []

        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue

            # Drop documents below similarity ratio threshold
            ratio = score / (top_score + 1e-8)
            if ratio < similarity_ratio_threshold:
                logger.debug(f"Filtered out rank {rank}: ratio={ratio:.3f} < {similarity_ratio_threshold}")
                break

            meta = self.metadata[idx]
            docs.append(RetrievedDocument(
                doc_id=meta["doc_id"],
                text=meta["text"],
                image_path=meta.get("image_path"),
                domain=meta["domain"],
                score=float(score),
                rank=rank,
            ))

        logger.debug(f"Retrieved {len(docs)}/{k} docs after adaptive-k filter (domain={self.domain})")
        return docs

    def format_context(self, docs: List[RetrievedDocument], max_chars: int = 1500) -> str:
        """
        Format retrieved documents into a context string for the LLM.

        Args:
            docs:      list of retrieved documents
            max_chars: maximum total context length

        Returns:
            Formatted context string
        """
        if not docs:
            return ""

        parts = []
        total_chars = 0

        for i, doc in enumerate(docs, start=1):
            header = f"[Reference {i} | similarity={doc.score:.3f}]"
            excerpt = doc.text[:max_chars // len(docs)]
            entry = f"{header}\n{excerpt}"

            if total_chars + len(entry) > max_chars:
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self):
        """Save FAISS index and metadata to disk."""
        if not FAISS_AVAILABLE:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f)
        logger.info(f"Vector store saved: {self.index_path}")

    def _load(self):
        """Load FAISS index and metadata from disk."""
        if not FAISS_AVAILABLE:
            return
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path) as f:
            self.metadata = json.load(f)
        logger.info(f"Vector store loaded: {self.domain}, {self.index.ntotal} vectors")


class MultiDomainRetriever:
    """
    Manages multiple MedicalVectorStore instances (one per domain).
    Selects the appropriate store based on DomainClassifier output.
    """

    def __init__(self, cfg: dict):
        self.cfg     = cfg
        self.stores  = {}
        self.max_k   = cfg["retrieval"]["max_k"]
        self.ratio_t = cfg["retrieval"]["similarity_ratio_threshold"]

        for domain in cfg["retrieval"]["domains"]:
            self.stores[domain] = MedicalVectorStore(
                domain=domain,
                embedding_dim=cfg["retrieval"]["embedding_dim"],
                store_dir=cfg["retrieval"]["vector_store_dir"],
            )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        domain: str,
        decision: str,           # from MeCoProbe.decide()
        min_k: int = 1,
    ) -> Tuple[List[RetrievedDocument], str]:
        """
        Retrieve documents for a given query.

        Adjusts max_k based on MeCo decision:
            SOFT_RAG → max_k = 1  (minimal retrieval)
            FULL_RAG → max_k = cfg.max_k

        Args:
            query_embedding: BiomedCLIP image embedding
            domain:          classified medical domain
            decision:        MeCoProbe decision string
            min_k:           minimum k regardless of filtering

        Returns:
            (list of RetrievedDocument, formatted_context_string)
        """
        from src.module1_metacognition.meco_probe import MeCoProbe

        if domain not in self.stores:
            logger.warning(f"Unknown domain '{domain}' — falling back to radiology")
            domain = "radiology"

        # Adjust k based on MeCo decision
        effective_max_k = 1 if decision == MeCoProbe.SOFT_RAG else self.max_k

        store = self.stores[domain]
        docs  = store.retrieve(
            query_embedding=query_embedding,
            max_k=effective_max_k,
            similarity_ratio_threshold=self.ratio_t,
        )

        context = store.format_context(docs)
        return docs, context
