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
  --gockpt-inflight-packets 64 \
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

## Visualize Power Metrics

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_power.py \
    --report benchmark/finetune_runs/power.json \
    --output benchmark/images/checkpoint_power.png
```

## Sweep Overlap Steps (Serial)

```bash
conda run -n checkpoint python benchmark/run_overlap_steps_benchmark.py \
  --overlap-steps 7,8,9,10,11,12,13,14 \
  --images-folder overlap_steps_7-14
```

This runs overlap steps in series and writes one combined line chart where
x-axis is algorithms and each overlap-step value is a separate line.

- `benchmark/images/overlap_steps_7-14/checkpoint_time_overlap_steps.png`

## Sanity Check

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 1000 \
  --save-steps 20 \
  --gradient-checkpointing
```
