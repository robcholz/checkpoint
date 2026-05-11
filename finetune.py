import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "yahma/alpaca-cleaned"
DEFAULT_OUTPUT_DIR = "checkpoints/qwen3-0.6b-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full finetuning for Qwen/Qwen3-0.6B on yahma/alpaca-cleaned."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        choices=(256, 512),
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size. Spec requires 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Training steps. Recommended range is 100 to 300.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10,
        choices=(10, 20),
        help="Checkpoint interval.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate for full finetuning.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log interval in steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM pressure.",
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "This script is configured for batch size 1 per the requested spec."
        )

    return args


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def format_prompt(example: Dict[str, str]) -> str:
    instruction = example["instruction"].strip()
    input_text = (example.get("input") or "").strip()

    if input_text:
        return (
            "Below is an instruction paired with an input. Write a helpful response.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )

    return (
        "Below is an instruction. Write a helpful response.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def build_preprocess_fn(tokenizer, max_length: int):
    def preprocess(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = format_prompt(example)
        response = example["output"].strip()

        full_text = f"{prompt}{response}{tokenizer.eos_token}"
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        prompt_len = min(len(prompt_tokens["input_ids"]), len(input_ids))

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This script targets a single RTX 4090.")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if supports_bf16() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = load_dataset(DATASET_NAME, split="train")
    preprocess_fn = build_preprocess_fn(tokenizer, args.seq_len)
    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
