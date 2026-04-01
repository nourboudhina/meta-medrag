"""
src/backbone/llava_med.py

LLaVA-Med backbone wrapper.
Handles model loading, image encoding, text generation,
and hidden-state extraction for the meta-cognition probe.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)
from loguru import logger


class LLaVAMedBackbone:
    """
    Thin wrapper around LLaVA-Med-1.5 (7B).

    Responsibilities:
        - Load model and tokeniser from HuggingFace / local checkpoint
        - Preprocess medical images
        - Run forward pass and return generated text
        - Expose internal hidden states at specified transformer layers
          (required by Module 1 — Meta-Cognition Probe)

    Usage:
        backbone = LLaVAMedBackbone(cfg)
        answer = backbone.generate(image, question)
        states  = backbone.get_hidden_states(image, question, layers=[-2,-5])
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.dtype  = torch.float16 if cfg.get("torch_dtype") == "float16" else torch.bfloat16
        self.model_name = cfg["model_name"]

        logger.info(f"Loading LLaVA-Med backbone: {self.model_name}")
        self._load_model()
        logger.info("Backbone loaded successfully")

    # ── Loading ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load tokeniser, image processor, and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)

        # 4-bit quantisation for inference on smaller GPUs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.model.eval()

    # ── Image preprocessing ────────────────────────────────────────────

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Load and preprocess a medical image into model-ready tensor.

        Args:
            image: file path or PIL Image

        Returns:
            pixel_values tensor of shape (1, C, H, W)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected str, Path, or PIL Image, got {type(image)}")

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        return pixel_values.to(self.device, dtype=self.dtype)

    # ── Prompt formatting ──────────────────────────────────────────────

    def format_prompt(self, question: str, retrieved_context: Optional[str] = None) -> str:
        """
        Build the LLaVA-Med prompt.

        LLaVA-Med uses a specific conversation template:
            USER: <image>\n{question} ASSISTANT:

        When retrieved context is available, it is prepended to the question.
        """
        if retrieved_context:
            user_content = (
                f"Reference information:\n{retrieved_context}\n\n"
                f"Based on the medical image and the reference above, {question}"
            )
        else:
            user_content = question

        prompt = (
            f"A chat between a curious user and an artificial intelligence "
            f"assistant specialising in medical imaging.\n"
            f"USER: <image>\n{user_content} ASSISTANT:"
        )
        return prompt

    # ── Generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        retrieved_context: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a clinical answer given image + question (+ optional retrieved context).

        Args:
            image:             medical image (path or PIL)
            question:          clinical question string
            retrieved_context: text from retrieved documents (Module 2 output)
            max_new_tokens:    generation limit

        Returns:
            Generated answer string
        """
        pixel_values = self.preprocess_image(image)
        prompt        = self.format_prompt(question, retrieved_context)

        input_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=self.cfg.get("do_sample", False),
            temperature=self.cfg.get("temperature", 0.1),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens (strip the prompt)
        generated = output_ids[0][input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer

    # ── Hidden state extraction ────────────────────────────────────────

    @torch.no_grad()
    def get_hidden_states(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
        layers: List[int] = [-2, -5, -8, -11, -15],
    ) -> Dict[int, np.ndarray]:
        """
        Extract internal hidden states at specified transformer layers.

        This is the core function used by Module 1 (Meta-Cognition Probe).
        Hidden states are extracted at the LAST token position — this captures
        the model's internal representation just before it generates an answer.

        Args:
            image:    medical image
            question: clinical question
            layers:   list of layer indices (negative = from the end)
                      e.g. [-2, -5, -8, -11, -15]

        Returns:
            dict mapping layer_index → numpy array of shape (hidden_dim,)
        """
        pixel_values = self.preprocess_image(image)
        prompt = self.format_prompt(question)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)

        # Forward pass with output_hidden_states=True
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        # each tensor has shape (batch, seq_len, hidden_dim)
        # We take the LAST token position as the representation
        all_hidden = outputs.hidden_states  # tuple: (embedding + N transformer layers)
        n_layers = len(all_hidden)

        extracted = {}
        for layer_idx in layers:
            # Convert negative index: -2 → n_layers - 2
            abs_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
            abs_idx = max(0, min(abs_idx, n_layers - 1))

            # Shape: (batch, seq_len, hidden_dim) → take last token → (hidden_dim,)
            hidden = all_hidden[abs_idx][0, -1, :].float().cpu().numpy()
            extracted[layer_idx] = hidden

        return extracted

    # ── Batch hidden states ────────────────────────────────────────────

    @torch.no_grad()
    def get_hidden_states_batch(
        self,
        items: List[Dict],           # list of {"image": ..., "question": ...}
        layers: List[int] = [-2, -5, -8, -11, -15],
        batch_size: int = 4,
    ) -> List[Dict[int, np.ndarray]]:
        """
        Extract hidden states for a list of (image, question) pairs.
        Used for training the meta-cognition probe.

        Args:
            items:      list of dicts with keys "image" and "question"
            layers:     which layers to extract
            batch_size: items per forward pass

        Returns:
            List of hidden-state dicts, one per item
        """
        all_states = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            for item in batch:
                states = self.get_hidden_states(
                    item["image"], item["question"], layers=layers
                )
                all_states.append(states)
            logger.debug(f"Extracted hidden states: {min(i+batch_size, len(items))}/{len(items)}")
        return all_states
