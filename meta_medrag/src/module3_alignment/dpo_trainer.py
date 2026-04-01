"""
src/module3_alignment/dpo_trainer.py

Direct Preference Optimization (DPO) fine-tuning for LLaVA-Med.
Uses LoRA for parameter-efficient training.

Based on:
    - DPO: Rafailov et al., NeurIPS 2023
    - MMed-RAG preference fine-tuning: Xia et al., ICLR 2025
    - RULE: Xia et al., 2024
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from datasets import Dataset

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("trl/peft not available — install with: pip install trl peft")


class MedRAGDPOTrainer:
    """
    DPO fine-tuning wrapper for LLaVA-Med.

    Loads the base model with 4-bit quantisation, applies LoRA adapters,
    then runs DPO training on the generated preference pairs.

    After training, the LoRA weights are merged into the base model
    and saved as a new checkpoint used for all inference.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.align_cfg = cfg["alignment"]
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

        if not TRL_AVAILABLE:
            raise ImportError("Install trl and peft: pip install trl peft bitsandbytes")

    def load_preference_data(self, data_path: str) -> Dataset:
        """
        Load and format preference pairs for DPO training.

        DPO trainer expects a HuggingFace Dataset with columns:
            - prompt:   the input question / instruction
            - chosen:   the preferred response
            - rejected: the dispreferred response

        Args:
            data_path: path to preference_pairs.json

        Returns:
            HuggingFace Dataset
        """
        with open(data_path) as f:
            raw = json.load(f)

        # Format: each item must have prompt, chosen, rejected
        formatted = []
        for item in raw:
            prompt   = item.get("prompt", item.get("question", ""))
            chosen   = item.get("chosen", "")
            rejected = item.get("rejected", "")

            if not all([prompt, chosen, rejected]):
                continue

            # Wrap in LLaVA-Med conversation format
            formatted.append({
                "prompt":   f"USER: {prompt} ASSISTANT:",
                "chosen":   chosen,
                "rejected": rejected,
            })

        dataset = Dataset.from_list(formatted)
        logger.info(f"DPO dataset: {len(dataset)} preference pairs")

        # Train/eval split
        split    = dataset.train_test_split(test_size=0.05, seed=42)
        return split

    def _load_model_and_tokenizer(self):
        """Load quantised model and tokeniser for LoRA fine-tuning."""
        model_name = self.align_cfg["base_model"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Prepare for k-bit training (required before applying LoRA)
        model = prepare_model_for_kbit_training(model)

        return model, tokenizer

    def _apply_lora(self, model):
        """Apply LoRA adapters to the model."""
        lora_cfg = LoraConfig(
            r=self.align_cfg["lora_r"],
            lora_alpha=self.align_cfg["lora_alpha"],
            lora_dropout=self.align_cfg["lora_dropout"],
            target_modules=self.align_cfg["lora_target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    def train(self, data_path: Optional[str] = None):
        """
        Run full DPO fine-tuning.

        Args:
            data_path: override for preference data path in config
        """
        data_path = data_path or self.align_cfg["preference_data_path"]
        output_dir = self.align_cfg["output_dir"]

        logger.info("Loading preference dataset...")
        split = self.load_preference_data(data_path)
        train_ds = split["train"]
        eval_ds  = split["test"]

        logger.info("Loading base model and applying LoRA...")
        model, tokenizer = self._load_model_and_tokenizer()
        model = self._apply_lora(model)

        # DPO training arguments
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=self.align_cfg["num_train_epochs"],
            per_device_train_batch_size=self.align_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=self.align_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=self.align_cfg["gradient_accumulation_steps"],
            learning_rate=self.align_cfg["learning_rate"],
            warmup_ratio=self.align_cfg["warmup_ratio"],
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            max_length=self.align_cfg["max_length"],
            max_prompt_length=512,
            beta=self.align_cfg["dpo_beta"],
            remove_unused_columns=False,
            report_to="none",      # set to "wandb" if you use W&B
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,        # implicit reference via PEFT (memory efficient)
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
        )

        logger.info("Starting DPO training...")
        trainer.train()

        # Save the fine-tuned model (LoRA weights only)
        logger.info(f"Saving LoRA checkpoint to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("DPO training complete!")
        return output_dir

    def merge_and_save(self, lora_path: str, output_path: str):
        """
        Merge LoRA adapters into the base model and save the full model.
        Use this to create a self-contained checkpoint for inference.

        Args:
            lora_path:   path to LoRA checkpoint (from train())
            output_path: where to save the merged model
        """
        from peft import PeftModel

        logger.info("Merging LoRA weights into base model...")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.align_cfg["base_model"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(lora_path)

        # Load and merge LoRA
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        merged     = peft_model.merge_and_unload()

        Path(output_path).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")


# ── Script entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    trainer = MedRAGDPOTrainer(cfg)
    trainer.train()
