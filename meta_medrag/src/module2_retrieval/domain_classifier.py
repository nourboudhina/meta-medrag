"""
src/module2_retrieval/domain_classifier.py

Domain identification using BiomedCLIP.

Given a medical image, classify it into one of:
    - radiology    (chest X-ray, CT, MRI)
    - pathology    (histology slides)
    - ophthalmology (fundus images)

This determines which FAISS vector store to query in Module 2.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from loguru import logger

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip not installed — domain classifier will use fallback")


# Text prompts defining each domain for zero-shot classification
DOMAIN_PROMPTS = {
    "radiology": [
        "a chest X-ray radiograph",
        "a CT scan of the chest",
        "an MRI scan",
        "a radiology image showing the thorax",
    ],
    "pathology": [
        "a histopathology slide under microscope",
        "a tissue biopsy slide",
        "a pathology image showing cells",
        "haematoxylin and eosin stained tissue",
    ],
    "ophthalmology": [
        "a fundus photograph of the eye",
        "a retinal image",
        "an optical coherence tomography scan of the retina",
        "an optic disc photograph",
    ],
}


class DomainClassifier:
    """
    Zero-shot medical image domain classifier using BiomedCLIP.

    Uses the CLIP image encoder + text encoder to compute cosine similarity
    between the input image embedding and domain-representative text prompts.
    The domain with the highest similarity is selected.

    This is the same approach used in MMed-RAG (Xia et al., ICLR 2025).
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.domains = list(DOMAIN_PROMPTS.keys())

        logger.info(f"Loading BiomedCLIP: {model_name}")
        self._load_model(model_name)
        self._precompute_text_embeddings()
        logger.info("DomainClassifier ready")

    def _load_model(self, model_name: str):
        """Load BiomedCLIP model and preprocessor."""
        if not OPEN_CLIP_AVAILABLE:
            self.model = None
            self.preprocess = None
            self.tokenizer  = None
            logger.warning("Using mock domain classifier (open_clip unavailable)")
            return

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _precompute_text_embeddings(self):
        """
        Pre-compute and cache text embeddings for all domain prompts.
        This is done once at init — not at every inference call.
        """
        if self.model is None:
            self._text_embeddings = {}
            return

        self._text_embeddings = {}
        for domain, prompts in DOMAIN_PROMPTS.items():
            tokens = self.tokenizer(prompts).to(self.device)
            embs   = self.model.encode_text(tokens)          # (n_prompts, dim)
            embs   = embs / embs.norm(dim=-1, keepdim=True)  # L2 normalise
            # Average across prompts for robust domain representation
            self._text_embeddings[domain] = embs.mean(dim=0)  # (dim,)

        logger.debug(f"Pre-computed text embeddings for {len(self._text_embeddings)} domains")

    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode a medical image into a CLIP embedding.

        Args:
            image: PIL Image, path string, or Path object

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.model is None:
            # Fallback: random embedding (for testing without open_clip)
            return np.random.randn(512).astype(np.float32)

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        emb    = self.model.encode_image(tensor)             # (1, dim)
        emb    = emb / emb.norm(dim=-1, keepdim=True)        # L2 normalise
        return emb[0].cpu().numpy()

    def classify(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Tuple[str, float, dict]:
        """
        Classify the medical domain of an image.

        Args:
            image: medical image (path or PIL)

        Returns:
            (domain_name, confidence_score, all_scores_dict)

        Example:
            domain, conf, scores = classifier.classify("chest_xray.jpg")
            # domain = "radiology", conf = 0.87
        """
        img_emb = self.encode_image(image)  # (dim,)

        scores = {}
        for domain, text_emb in self._text_embeddings.items():
            if isinstance(text_emb, torch.Tensor):
                text_np = text_emb.cpu().numpy()
            else:
                text_np = text_emb

            # Cosine similarity (both vectors are already L2-normalised)
            sim = float(np.dot(img_emb, text_np))
            scores[domain] = sim

        # Apply softmax for calibrated probabilities
        vals   = np.array(list(scores.values()))
        softmax = np.exp(vals - vals.max()) / np.exp(vals - vals.max()).sum()
        probs   = {d: float(p) for d, p in zip(scores.keys(), softmax)}

        best_domain = max(probs, key=probs.get)
        confidence  = probs[best_domain]

        logger.debug(f"Domain: {best_domain} ({confidence:.2%}) | scores: {probs}")
        return best_domain, confidence, probs

    def classify_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
    ) -> List[Tuple[str, float, dict]]:
        """Classify a list of images."""
        return [self.classify(img) for img in images]
