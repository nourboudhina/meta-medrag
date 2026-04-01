"""
src/module1_metacognition/contrastive_dataset.py

Builds the contrastive pair dataset used to train the MeCo probe.

Strategy:
    KNOWN (label=0):   Common findings the model consistently answers correctly.
                       Source: MIMIC-CXR reports, well-known radiology patterns.
    UNKNOWN (label=1): Rare/recent findings the model fails on.
                       Generated with GPT-4o to create hard adversarial questions.

Output JSON format:
    [{"image": "...", "question": "...", "label": 0|1, "domain": "radiology"}, ...]
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI


# Seed questions that models reliably know (label=0)
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

# Prompt to GPT-4o for generating hard (unknown) questions
HARD_QUESTION_PROMPT = """You are a medical AI evaluation expert.
Generate {n} challenging radiology questions that a 7B medical vision-language model 
is likely to answer INCORRECTLY. These should involve:
- Rare diseases or uncommon presentations
- Subtle findings that require expert-level knowledge  
- Recent clinical guidelines from 2023-2024 that may not be in training data
- Multi-finding questions requiring complex reasoning

Format: return ONLY a JSON list of question strings, no other text.
Example: ["Is this pattern consistent with organizing pneumonia rather than COVID-19?", ...]
"""


class ContrastiveDatasetBuilder:
    """
    Builds a balanced contrastive dataset for probe training.

    Usage:
        builder = ContrastiveDatasetBuilder(openai_api_key="sk-...")
        builder.build(
            known_images=list_of_image_paths,
            output_path="data/processed/contrastive_pairs.json",
            n_pairs=500,
        )
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)

        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None
            logger.warning("No OpenAI key — hard questions will use template fallbacks")

    # ── Known samples ─────────────────────────────────────────────────

    def generate_known_samples(
        self,
        image_paths: List[str],
        n: int = 250,
    ) -> List[Dict]:
        """
        Create label=0 samples using standard radiology questions.
        These are questions the model reliably knows from training.
        """
        samples = []
        images  = random.choices(image_paths, k=n)

        for img in images:
            q = random.choice(KNOWN_QUESTION_TEMPLATES)
            samples.append({
                "image":    str(img),
                "question": q,
                "label":    0,
                "domain":   "radiology",
            })

        logger.info(f"Generated {len(samples)} known samples (label=0)")
        return samples

    # ── Unknown samples ───────────────────────────────────────────────

    def generate_unknown_samples_gpt4o(
        self,
        image_paths: List[str],
        n: int = 250,
        batch_size: int = 20,
    ) -> List[Dict]:
        """
        Create label=1 samples using GPT-4o to generate hard questions.
        Requires OpenAI API key.
        """
        if self.client is None:
            return self._generate_unknown_fallback(image_paths, n)

        questions = []
        while len(questions) < n:
            needed = min(batch_size, n - len(questions))
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": HARD_QUESTION_PROMPT.format(n=needed)
                    }],
                    max_tokens=1024,
                    temperature=0.8,
                )
                raw = resp.choices[0].message.content.strip()
                batch = json.loads(raw)
                questions.extend(batch[:needed])
                logger.debug(f"Generated {len(questions)}/{n} unknown questions")
            except Exception as e:
                logger.warning(f"GPT-4o error: {e} — using fallback")
                questions.extend(self._fallback_hard_questions(needed))

        samples = []
        images  = random.choices(image_paths, k=len(questions))
        for img, q in zip(images, questions[:n]):
            samples.append({
                "image":    str(img),
                "question": q,
                "label":    1,
                "domain":   "radiology",
            })

        logger.info(f"Generated {len(samples)} unknown samples (label=1)")
        return samples

    def _generate_unknown_fallback(self, image_paths: List[str], n: int) -> List[Dict]:
        """Fallback hard questions when OpenAI is not available."""
        hard_qs = self._fallback_hard_questions(n)
        images  = random.choices(image_paths, k=n)
        return [{"image": str(img), "question": q, "label": 1, "domain": "radiology"}
                for img, q in zip(images, hard_qs)]

    def _fallback_hard_questions(self, n: int) -> List[str]:
        templates = [
            "Does this chest CT show features of hypersensitivity pneumonitis rather than usual interstitial pneumonia?",
            "Is the pattern consistent with respiratory bronchiolitis-ILD versus desquamative interstitial pneumonia?",
            "Does this image show signs consistent with the 2023 ATS criteria for progressive pulmonary fibrosis?",
            "Are the findings indicative of pulmonary veno-occlusive disease rather than pulmonary arterial hypertension?",
            "Does this radiograph show features of acute fibrinous and organising pneumonia (AFOP)?",
        ]
        return [random.choice(templates) for _ in range(n)]

    # ── Build full dataset ────────────────────────────────────────────

    def build(
        self,
        image_paths: List[str],
        output_path: str,
        n_known: int = 250,
        n_unknown: int = 250,
    ) -> str:
        """
        Build and save the full contrastive dataset.

        Returns path to saved JSON.
        """
        known   = self.generate_known_samples(image_paths, n=n_known)
        unknown = self.generate_unknown_samples_gpt4o(image_paths, n=n_unknown)

        dataset = known + unknown
        random.shuffle(dataset)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Contrastive dataset saved: {len(dataset)} items → {output_path}")
        return output_path
