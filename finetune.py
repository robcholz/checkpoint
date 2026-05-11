import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from src.baseline_hook import BaselineCheckpointConfig, BaselineCheckpointHook

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
        "--weight-decay", type=float, default=0.01, help="AdamW weight decay."
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--logging-steps", type=int, default=1, help="Log interval in steps."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and final model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
            full_text, truncation=True, max_length=max_length, padding=False
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


def cycle_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def get_train_dtype() -> torch.dtype:
    return torch.bfloat16 if supports_bf16() else torch.float16


def save_training_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    hook: BaselineCheckpointHook,
    losses: List[dict],
) -> None:
    metadata = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "checkpoint_files": [str(result.path) for result in hook.history],
        "checkpoint_durations_sec": [result.duration_sec for result in hook.history],
        "loss_history": losses,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. This script targets a single RTX 4090.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hook_checkpoint_dir = output_dir / "baseline_hook"
    hook_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dtype = get_train_dtype()
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=train_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup_steps = math.ceil(args.max_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    hook = BaselineCheckpointHook(
        model=model,
        optimizer=optimizer,
        config=BaselineCheckpointConfig(
            checkpoint_dir=hook_checkpoint_dir,
            tag_prefix="baseline_step",
            save_model=True,
            save_optimizer=True,
            save_rng_state=True,
        ),
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    preprocess_fn = build_preprocess_fn(tokenizer, args.seq_len)
    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SupervisedDataCollator(tokenizer),
        pin_memory=True,
    )

    scaler = GradScaler("cuda", enabled=train_dtype == torch.float16)
    batch_iterator = cycle_dataloader(train_dataloader)
    loss_history: List[dict] = []
    train_start = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        if step % args.save_steps == 0:
            hook.save_checkpoint(step)

        batch = next(batch_iterator)
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)

        hook.backward_begin(step)
        with autocast("cuda", dtype=train_dtype):
            outputs = model(**batch)
            loss = outputs.loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        hook.backward_end(step)

        hook.update_begin(step)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        hook.update_end(step)
        scheduler.step()

        step_loss = float(loss.detach().item())
        current_lr = float(scheduler.get_last_lr()[0])
        loss_history.append(
            {"step": step, "loss": step_loss, "learning_rate": current_lr}
        )

        if step % args.logging_steps == 0:
            elapsed = time.perf_counter() - train_start
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            print(
                f"step={step}/{args.max_steps} "
                f"loss={step_loss:.4f} "
                f"lr={current_lr:.8f} "
                f"steps_per_sec={steps_per_sec:.2f}",
                flush=True,
            )

    total_runtime = time.perf_counter() - train_start
    print(f"train_runtime_sec={total_runtime:.2f}", flush=True)
    print(f"train_steps_per_sec={args.max_steps / total_runtime:.2f}", flush=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_training_metadata(args, output_dir, hook, loss_history)


if __name__ == "__main__":
    main()
