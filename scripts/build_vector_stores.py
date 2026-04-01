"""
scripts/build_vector_stores.py

Build FAISS vector stores for all medical domains.

Run once before starting inference:
    python scripts/build_vector_stores.py --domain radiology
    python scripts/build_vector_stores.py --domain all
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from loguru import logger

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


def get_text_encoder(model_name: str, device: str):
    """Return a callable that encodes text to BiomedCLIP embeddings."""
    if not OPEN_CLIP_AVAILABLE:
        logger.warning("open_clip not available — using random embeddings (for testing)")
        def mock_encoder(text: str) -> np.ndarray:
            return np.random.randn(512).astype(np.float32)
        return mock_encoder

    model, _, _ = open_clip.create_model_and_transforms(
        f"hf-hub:{model_name}"
    )
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_name}")
    model = model.to(device).eval()

    @torch.no_grad()
    def encode_text(text: str) -> np.ndarray:
        tokens = tokenizer([text]).to(device)
        emb    = model.encode_text(tokens)[0]
        emb    = emb / emb.norm()
        return emb.cpu().numpy().astype(np.float32)

    return encode_text


def load_reports(dataset_cfg: dict, domain: str, limit: int = None) -> list:
    """
    Load reports from a dataset and return as list of dicts.
    Format: [{"doc_id": ..., "text": ..., "image_path": ...}]
    """
    report_dir = Path(dataset_cfg["report_dir"])
    image_dir  = Path(dataset_cfg["image_dir"])

    reports = []
    report_files = list(report_dir.glob("*.txt"))[:limit]

    for rf in report_files:
        with open(rf) as f:
            text = f.read().strip()

        # Find matching image
        img_path = image_dir / rf.stem
        for ext in [".jpg", ".png", ".dcm"]:
            if (img_path.with_suffix(ext)).exists():
                img_path = str(img_path.with_suffix(ext))
                break
        else:
            img_path = None

        reports.append({
            "doc_id":     rf.stem,
            "text":       text,
            "image_path": str(img_path) if img_path else None,
            "domain":     domain,
        })

    logger.info(f"Loaded {len(reports)} reports for domain={domain}")
    return reports


def build_domain_store(cfg: dict, domain: str, device: str):
    """Build and save the FAISS index for one domain."""
    from src.module2_retrieval.vector_store import MedicalVectorStore

    ret_cfg = cfg["retrieval"]

    # Map domain to dataset
    domain_to_dataset = {
        "radiology":    "iu_xray",
        "pathology":    "iu_xray",
        "ophthalmology":"iu_xray",
    }
    dataset_name = domain_to_dataset.get(domain, "mimic_cxr")
    dataset_cfg  = cfg["datasets"].get(dataset_name, {})

    if not dataset_cfg:
        logger.warning(f"No dataset config for domain={domain}")
        return

    reports = load_reports(dataset_cfg, domain)
    if not reports:
        logger.warning(f"No reports found for domain={domain}")
        return

    text_encoder = get_text_encoder(ret_cfg["biomed_clip_model"], device)

    store = MedicalVectorStore(
        domain=domain,
        embedding_dim=ret_cfg["embedding_dim"],
        store_dir=ret_cfg["vector_store_dir"],
    )
    store.build(reports=reports, text_encoder=text_encoder)
    logger.info(f"Vector store built for domain={domain}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all",
                        choices=["radiology", "pathology", "ophthalmology", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    domains = cfg["retrieval"]["domains"] if args.domain == "all" else [args.domain]

    for domain in domains:
        logger.info(f"Building vector store for domain: {domain}")
        build_domain_store(cfg, domain, args.device)

    logger.info("All vector stores built successfully")


if __name__ == "__main__":
    main()
