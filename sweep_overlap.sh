conda run -n checkpoint python benchmark/run_overlap_steps_benchmark.py \
  --overlap-steps 2,3,4,5,6,7 \
  --gockpt-transfer-chunk-mb 64 \
  --output-dir /tmp/zsheng1/overlap_steps_runs \
  --images-folder overlap_steps_2-7_chunk64
