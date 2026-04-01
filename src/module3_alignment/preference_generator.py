"""
src/module3_alignment/preference_generator.py

Generates DPO preference pairs for cross-modal alignment fine-tuning.

Three types of preference pairs (from MMed-RAG + RULE):
    Type 1 — Cross-modal alignment:
        chosen:   response that correctly uses the medical image
        rejected: response that ignores the image, relies only on retrieved text

    Type 2 — Over-reliance mitigation:
        chosen:   model's correct internal answer (no RAG)
        rejected: model's wrong answer caused by noisy retrieved context

    Type 3 — Knowledge balance:
        chosen:   response that integrates image + retrieved context correctly
        rejected: response that uses retrieved context but misses image findings
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI


PAIR_TYPE1_SYSTEM = """You are a medical AI evaluation expert and dataset creator.
Your task is to create preference pairs for training a medical vision-language model 
to correctly use medical images rather than ignoring them.

Given a medical image description and a clinical question, generate:
- chosen: a response that grounds the answer in specific visual findings from the image
- rejected: a response that ignores the image and relies only on general medical knowledge 
  or retrieved text, potentially hallucinating findings not visible in the image

The chosen response MUST reference specific visual features (e.g., "the X-ray shows blunting 
of the left costophrenic angle", "the consolidation is visible in the right lower lobe").
The rejected response should sound plausible but NOT reference the specific image.

Return ONLY valid JSON: {"chosen": "...", "rejected": "..."}"""

PAIR_TYPE2_SYSTEM = """You are a medical AI evaluation expert.
Given a medical report and a clinical question, create a preference pair showing:
- chosen: a correct, concise answer based on the actual medical findings
- rejected: an incorrect answer that results from over-relying on irrelevant 
  retrieved context that led the model astray

Return ONLY valid JSON: {"chosen": "...", "rejected": "..."}"""

PAIR_TYPE3_SYSTEM = """You are a medical AI evaluation expert.
Given a radiology report and a clinical question, create a preference pair showing:
- chosen: a response that correctly integrates the image findings with the retrieved 
  reference information, prioritising the actual image evidence
- rejected: a response that correctly uses the retrieved reference but fails to 
  account for patient-specific findings visible in the actual image

Return ONLY valid JSON: {"chosen": "...", "rejected": "..."}"""


class PreferenceGenerator:
    """
    Generates DPO preference pairs using GPT-4o as a synthetic data creator.
    These pairs are used to fine-tune LLaVA-Med with Direct Preference Optimization.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model  = model

    def _call_gpt4o(self, system: str, user: str) -> Optional[Dict]:
        """Call GPT-4o and parse JSON response."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=1024,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"GPT-4o call failed: {e}")
            return None

    def generate_type1_pair(
        self,
        image_description: str,
        question: str,
        report_text: str,
    ) -> Optional[Dict]:
        """Generate a cross-modal alignment preference pair."""
        user = (
            f"Medical image description: {image_description}\n\n"
            f"Clinical question: {question}\n\n"
            f"Radiology report (for context): {report_text[:500]}"
        )
        result = self._call_gpt4o(PAIR_TYPE1_SYSTEM, user)
        if result and "chosen" in result and "rejected" in result:
            return {"type": 1, "question": question, **result}
        return None

    def generate_type2_pair(
        self,
        question: str,
        correct_answer: str,
        noisy_context: str,
    ) -> Optional[Dict]:
        """Generate an over-reliance mitigation preference pair."""
        user = (
            f"Clinical question: {question}\n\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Noisy retrieved context that may mislead: {noisy_context[:400]}"
        )
        result = self._call_gpt4o(PAIR_TYPE2_SYSTEM, user)
        if result and "chosen" in result and "rejected" in result:
            return {"type": 2, "question": question, **result}
        return None

    def generate_type3_pair(
        self,
        question: str,
        report_text: str,
        retrieved_reference: str,
    ) -> Optional[Dict]:
        """Generate a knowledge-balance preference pair."""
        user = (
            f"Clinical question: {question}\n\n"
            f"Patient report (image findings): {report_text[:500]}\n\n"
            f"Retrieved reference: {retrieved_reference[:400]}"
        )
        result = self._call_gpt4o(PAIR_TYPE3_SYSTEM, user)
        if result and "chosen" in result and "rejected" in result:
            return {"type": 3, "question": question, **result}
        return None

    def generate_from_dataset(
        self,
        reports: List[Dict],
        output_path: str,
        n_per_report: int = 3,
    ) -> str:
        """
        Generate all three types of preference pairs from a list of reports.

        Args:
            reports:       list of dicts with "text", "question", "image_path"
            output_path:   where to save the preference dataset JSON
            n_per_report:  number of pairs per report

        Returns:
            path to saved dataset
        """
        all_pairs = []

        for i, doc in enumerate(reports):
            q    = doc.get("question", "Describe the findings in this medical image.")
            text = doc.get("text", "")
            img  = doc.get("image_path", "")

            # Type 1: cross-modal alignment
            p1 = self.generate_type1_pair(
                image_description=f"Medical image: {img}",
                question=q,
                report_text=text,
            )
            if p1:
                p1["image"]  = img
                p1["prompt"] = q
                all_pairs.append(p1)

            # Type 3: knowledge balance (if retrieved context available)
            if "retrieved_context" in doc:
                p3 = self.generate_type3_pair(
                    question=q,
                    report_text=text,
                    retrieved_reference=doc["retrieved_context"],
                )
                if p3:
                    p3["image"]  = img
                    p3["prompt"] = q
                    all_pairs.append(p3)

            if (i + 1) % 20 == 0:
                logger.info(f"Generated {len(all_pairs)} pairs from {i+1}/{len(reports)} reports")

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_pairs, f, indent=2)

        type_counts = {t: sum(1 for p in all_pairs if p["type"] == t) for t in [1, 2, 3]}
        logger.info(f"Saved {len(all_pairs)} preference pairs → {output_path}")
        logger.info(f"Type distribution: {type_counts}")
        return output_path
