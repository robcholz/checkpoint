# Checkpoint

## Verify

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 200 \
  --save-steps 20 \
  --gradient-checkpointing
```

## Checkpoint Benchmark

```bash
conda run -n checkpoint python benchmark/checkpoint_benchmark.py \
  --steps 10 \
  --save-step 3 \
  --overlap-steps 4 \
  --output-dir benchmark/checkpoint_runs
```

The benchmark runs `baseline`, `async`, `async_o`, `gockpt`, and `gockpt_o`
on a deterministic tiny model. It verifies that each checkpoint loads, resumes,
and reaches the same final model and optimizer state as uninterrupted training.
It also writes phase timing summaries and expectation checks to
`benchmark/checkpoint_runs/report.json`.

## Real Finetune Benchmark

```bash
conda run -n checkpoint python benchmark/finetune_benchmark.py \
  --hook-types baseline async async_o gockpt gockpt_o \
  --seq-len 512 \
  --max-steps 200 \
  --save-steps 20 \
  --overlap-steps 7 \
  --gradient-checkpointing \
  --output-dir benchmark/finetune_runs
```

This benchmark runs the real `finetune.py` Qwen training loop once per hook,
enables phase profiling, skips the final Hugging Face model save by default,
and writes `benchmark/finetune_runs/report.json`. The report compares
throughput, wall time, checkpoint count, and hook-phase stall time. Use this
benchmark for real checkpointing performance; use `checkpoint_benchmark.py`
for fast correctness and resume validation.

The runner checks GPU 0 free memory before launching runs. If another process
is using the RTX 4090, free the GPU first or lower `--min-free-gb` only for a
constrained smoke run.

## Finetuning

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 1000 \
  --save-steps 20 \
  --gradient-checkpointing
```
