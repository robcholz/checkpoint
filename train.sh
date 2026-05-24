conda run -n checkpoint python benchmark/finetune_benchmark.py \
  --hook-types baseline async async_o gockpt gockpt_o \
  --seq-len 512 \
  --max-steps 207 \
  --save-steps 50 \
  --overlap-steps 2 \
  --gockpt-inflight-packets 64 \
  --gockpt-transfer-chunk-mb 64 \
  --gradient-checkpointing \
  --output-dir /tmp/zsheng1/finetune_runs

rsync -avh --delete --progress --exclude='*.pt' /tmp/zsheng1/finetune_runs/ benchmark/finetune_runs/

conda run -n checkpoint python benchmark/visualize_checkpoint_power.py \
    --report /tmp/zsheng1/finetune_runs/power.json \
    --output benchmark/images/checkpoint_power.png

conda run -n checkpoint python benchmark/visualize_checkpoint_save_time.py \
    --report /tmp/zsheng1/finetune_runs/report.json \
    --output benchmark/images/foreground_checkpoint_time.png \
    --title "Foreground Checkpoint Stall Time vs Algorithms"

conda run -n checkpoint python benchmark/visualize_checkpoint_memory.py \
    --report benchmark/finetune_runs/host_memory.json \
    --output benchmark/images/checkpoint_memory.png \
    --title "Host Memory Usage Over Time"

conda run -n checkpoint python benchmark/visualize_checkpoint_phase.py  \
    --report benchmark/finetune_runs/report.json \
    -o benchmark/images/checkpoint_phase.png
