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
  --overlap-steps 10 \
  --gradient-checkpointing \
  --output-dir benchmark/finetune_runs
```

## Benchmark Image

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_save_time.py \
    --report benchmark/finetune_runs/report.json \
    --output benchmark/images/foreground_checkpoint_time.png \
    --title "Foreground Checkpoint Stall Time vs Algorithms"
```

## Sweep Overlap Steps (Serial)

```bash
conda run -n checkpoint python benchmark/run_overlap_steps_benchmark.py \
  --overlap-steps 7,8,9,10 \
  --images-folder overlap_steps_7_10
```

This runs each overlap step in series (no parallel execution), and generates:

- `benchmark/images/overlap_steps_7_10/checkpoint_time_overlap_steps=7.png`
- `benchmark/images/overlap_steps_7_10/checkpoint_time_overlap_steps=8.png`
- `benchmark/images/overlap_steps_7_10/checkpoint_time_overlap_steps=9.png`
- `benchmark/images/overlap_steps_7_10/checkpoint_time_overlap_steps=10.png`

## Sanity Check

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 1000 \
  --save-steps 20 \
  --gradient-checkpointing
```
