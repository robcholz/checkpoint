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
  --max-steps 207 \
  --save-steps 50 \
  --overlap-steps 7 \
  --gockpt-inflight-packets 64 \
  --gockpt-transfer-chunk-mb 64 \
  --gradient-checkpointing \
  --output-dir /tmp/zsheng1/finetune_runs
```

Use `--gockpt-transfer-chunk-mb 64` for the primary run because the local PCIe
sweep gave the best GoCkpt-O throughput at 64 MiB. Re-run the PCIe sweep below
if you need to re-measure the best chunk size on a different machine.

## Benchmark Image

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_save_time.py \
    --report /tmp/zsheng1/finetune_runs/report.json \
    --output benchmark/images/foreground_checkpoint_time.png \
    --title "Foreground Checkpoint Stall Time vs Algorithms"
```

## Visualize Power Metrics

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_power.py \
    --report /tmp/zsheng1/finetune_runs/power.json \
    --output benchmark/images/checkpoint_power.png
```

## Visualize Memory Metrics

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_memory.py \
    --report benchmark/finetune_runs/host_memory.json \
    --output benchmark/images/checkpoint_memory.png \
    --title "Host Memory Usage Over Time"
```

## Visualize Phase

```bash
conda run -n checkpoint python benchmark/visualize_checkpoint_phase.py  \
    --report benchmark/finetune_runs/report.json \
    -o benchmark/images/checkpoint_phase.png
```

## Copy

```bash
rsync -avh --progress --exclude='*.pt' /tmp/zsheng1/finetune_runs/ benchmark/finetune_runs/
```

## Sweep Overlap Steps (Serial)

```bash
conda run -n checkpoint python benchmark/run_overlap_steps_benchmark.py \
  --overlap-steps 2,3,4,5,6,7 \
  --gockpt-transfer-chunk-mb 64 \
  --images-folder overlap_steps_2-7_chunk64
```

## Sweep PCIe Steps (Serial)

```bash
conda run -n checkpoint python benchmark/run_pcie_benchmark.py \
  --hook-types gockpt gockpt_o \
  --seq-len 512 \
  --max-steps 200 \
  --save-steps 20 \
  --overlap-steps 7 \
  --gockpt-inflight-packets 64 \
  --gradient-checkpointing \
  --transfer-chunk-mb 0,16,32,64,128 \
  --images-folder transfer_chunk_0-128
```

## Visualize Ringbuffer Pressure

```bash
conda run -n checkpoint python benchmark/run_ringbuffer_pressure.py \
  --hook-types gockpt gockpt_o \
  --seq-len 512 \
  --max-steps 200 \
  --save-steps 20 \
  --overlap-steps 7 \
  --gockpt-inflight-packets 64,128,256 \
  --gradient-checkpointing \
  --images-folder ringbuffer_pressure_64-256
```

## Sanity Check

```bash
python finetune.py \
  --seq-len 512 \
  --max-steps 1000 \
  --save-steps 20 \
  --gradient-checkpointing
```
