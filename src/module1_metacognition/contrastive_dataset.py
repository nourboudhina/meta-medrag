"""
src/module1_metacognition/contrastive_dataset.py

Builds contrastive pairs from ALL available datasets:
    - IU-Xray       → known samples (common radiology findings)
    - VQA-RAD       → known samples (simple yes/no questions)
    - SLAKE         → unknown samples (complex open questions)
    - MIMIC-CXR     → known samples (if available)
    Hard templates  → unknown samples (rare diseases)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

HARD_QUESTION_TEMPLATES = [
    "Does this chest CT show features of hypersensitivity pneumonitis rather than usual interstitial pneumonia?",
    "Is the pattern consistent with respiratory bronchiolitis-ILD versus desquamative interstitial pneumonia?",
    "Does this image show signs consistent with the 2023 ATS criteria for progressive pulmonary fibrosis?",
    "Are the findings indicative of pulmonary veno-occlusive disease rather than pulmonary arterial hypertension?",
    "Does this radiograph show features of acute fibrinous and organising pneumonia (AFOP)?",
    "Are these findings consistent with lymphangioleiomyomatosis (LAM)?",
    "Does this show evidence of combined pulmonary fibrosis and emphysema syndrome?",
    "Is this pattern consistent with non-specific interstitial pneumonia (NSIP)?",
    "Does this chest CT show tree-in-bud opacities suggesting endobronchial spread of tuberculosis?",
    "Does this show features of yellow nail syndrome with pleural effusion?",
    "Are the findings consistent with pulmonary alveolar proteinosis?",
    "Is there evidence of amyloidosis affecting the lungs in this image?",
    "Does this CT show features of pleuroparenchymal fibroelastosis (PPFE)?",
    "Does this image show Bochdalek hernia in an adult presentation?",
    "Are the findings consistent with Swyer-James-MacLeod syndrome?",
]

KNOWN_QUESTION_TEMPLATES = [
    "Is there evidence of cardiomegaly in this chest X-ray?",
    "Are the lung fields clear in this image?",
    "Is the cardiac silhouette within normal limits?",
    "Does this image show bilateral pleural effusion?",
    "Are the costophrenic angles sharp?",
    "Is there any evidence of pneumothorax?",
    "Does this chest X-ray show pulmonary oedema?",
    "Are the bony structures intact in this image?",
    "Is the trachea midline in this chest X-ray?",
    "Does this image show consolidation?",
]


class ContrastiveDatasetBuilder:
    """
    Builds balanced contrastive dataset from all available datasets.
    Uses IU-Xray + VQA-RAD + SLAKE + MIMIC-CXR (optional).
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)
        self.client = None
        if openai_api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client ready — will generate hard questions with GPT-4o")
        else:
            logger.warning("No OpenAI key — using hard question templates as fallback")

    def build_from_all_datasets(
        self,
        output_path: str = "data/processed/contrastive_pairs.json",
        n_per_source: int = 150,
    ) -> str:
        """
        Build contrastive pairs from ALL available datasets.
        This is the main method to call.
        """
        all_known   = []
        all_unknown = []

        # ── Source 1: IU-Xray → KNOWN ─────────────────────────────
        iu_path = Path("data/raw/iu_xray/splits.json")
        if iu_path.exists():
            iu_data  = json.load(open(iu_path, encoding="utf-8"))
            iu_train = [x for x in iu_data
                        if x.get("split") == "train" and x.get("image")]
            sample   = random.sample(iu_train, min(n_per_source, len(iu_train)))
            for item in sample:
                all_known.append({
                    "image":    item["image"],
                    "question": item.get("question",
                                random.choice(KNOWN_QUESTION_TEMPLATES)),
                    "label":    0,
                    "domain":   "radiology",
                    "source":   "iu_xray",
                })
            logger.info(f"IU-Xray known: {len(sample)} samples")

        # ── Source 2: VQA-RAD → KNOWN ─────────────────────────────
        vqa_path = Path("data/raw/vqa_rad/splits.json")
        if vqa_path.exists():
            vqa_data  = json.load(open(vqa_path, encoding="utf-8"))
            vqa_train = [x for x in vqa_data
                         if x.get("split") == "train" and x.get("image")]
            sample    = random.sample(vqa_train, min(n_per_source, len(vqa_train)))
            for item in sample:
                all_known.append({
                    "image":    item["image"],
                    "question": item.get("question", ""),
                    "label":    0,
                    "domain":   "radiology",
                    "source":   "vqa_rad",
                })
            logger.info(f"VQA-RAD known: {len(sample)} samples")

        # ── Source 3: MIMIC-CXR → KNOWN (si disponible) ───────────
        mimic_path = Path("data/raw/mimic_cxr/splits.json")
        if mimic_path.exists():
            mimic_data  = json.load(open(mimic_path, encoding="utf-8"))
            mimic_train = [x for x in mimic_data
                           if x.get("split") == "train" and x.get("image")]
            if mimic_train:
                sample = random.sample(
                    mimic_train, min(50, len(mimic_train))
                )
                for item in sample:
                    all_known.append({
                        "image":    item["image"],
                        "question": item.get("question",
                                    random.choice(KNOWN_QUESTION_TEMPLATES)),
                        "label":    0,
                        "domain":   "radiology",
                        "source":   "mimic_cxr",
                    })
                logger.info(f"MIMIC-CXR known: {len(sample)} samples")

        # ── Source 4: SLAKE open questions → UNKNOWN ──────────────
        slake_path = Path("data/raw/slake/splits.json")
        if slake_path.exists():
            slake_data  = json.load(open(slake_path, encoding="utf-8"))
            slake_train = [x for x in slake_data
                           if x.get("split") == "train"]
            # Questions ouvertes = plus difficiles = unknown
            open_q = [x for x in slake_train
                      if str(x.get("q_type", "")).upper() != "CLOSED"]
            sample = random.sample(open_q, min(n_per_source, len(open_q)))

            # Utilise les images IU-Xray pour SLAKE (SLAKE n'a pas d'images locales)
            iu_images = []
            if iu_path.exists():
                iu_all    = json.load(open(iu_path, encoding="utf-8"))
                iu_images = [x["image"] for x in iu_all if x.get("image")]

            for i, item in enumerate(sample):
                img = item.get("image") or (
                    iu_images[i % len(iu_images)] if iu_images else None
                )
                all_unknown.append({
                    "image":    img,
                    "question": item.get("question", ""),
                    "label":    1,
                    "domain":   "radiology",
                    "source":   "slake",
                })
            logger.info(f"SLAKE unknown: {len(sample)} samples")

        # ── Source 5: Hard templates → UNKNOWN ────────────────────
        if self.client:
            hard_qs = self._generate_gpt4o_questions(n_per_source)
        else:
            hard_qs = self._template_hard_questions(n_per_source)

        iu_images = []
        if iu_path.exists():
            iu_all    = json.load(open(iu_path, encoding="utf-8"))
            iu_images = [x["image"] for x in iu_all if x.get("image")]

        for i, q in enumerate(hard_qs):
            img = iu_images[i % len(iu_images)] if iu_images else None
            all_unknown.append({
                "image":    img,
                "question": q,
                "label":    1,
                "domain":   "radiology",
                "source":   "hard_template",
            })
        logger.info(f"Hard unknown: {len(hard_qs)} samples")

        # ── Balance + shuffle ──────────────────────────────────────
        n      = min(len(all_known), len(all_unknown))
        known  = random.sample(all_known,   n)
        unk    = random.sample(all_unknown, n)
        dataset = known + unk
        random.shuffle(dataset)

        # ── Save ───────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        sources = {}
        for x in dataset:
            s = x.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1

        logger.info(f"Contrastive dataset: {len(dataset)} pairs")
        logger.info(f"  Known (0):   {sum(1 for x in dataset if x['label']==0)}")
        logger.info(f"  Unknown (1): {sum(1 for x in dataset if x['label']==1)}")
        logger.info(f"  Sources: {sources}")
        return output_path

    def _template_hard_questions(self, n: int) -> List[str]:
        result = []
        while len(result) < n:
            result.extend(HARD_QUESTION_TEMPLATES)
        return random.sample(result, n)

    def _generate_gpt4o_questions(self, n: int) -> List[str]:
        questions = []
        batch_size = 20
        while len(questions) < n:
            needed = min(batch_size, n - len(questions))
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content":
                        f"Generate {needed} challenging radiology questions "
                        f"that a medical AI model is likely to answer incorrectly. "
                        f"Focus on rare diseases, subtle findings, recent guidelines. "
                        f"Return ONLY a JSON list of strings."}],
                    max_tokens=1024,
                    temperature=0.8,
                )
                batch = json.loads(resp.choices[0].message.content.strip())
                questions.extend(batch[:needed])
            except Exception as e:
                logger.warning(f"GPT-4o error: {e} — using templates")
                questions.extend(self._template_hard_questions(needed))
        return questions[:n]

    # ── Backward compatibility ─────────────────────────────────────
    def build(
        self,
        image_paths: List[str],
        output_path: str,
        n_known: int = 250,
        n_unknown: int = 250,
    ) -> str:
        return self.build_from_all_datasets(output_path=output_path)