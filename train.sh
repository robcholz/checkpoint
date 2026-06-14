case "${1-}" in
  "" | --*)
    benchmark=benchmark
    ;;
  *)
    benchmark=$1
    shift
    ;;
esac
benchmark=${benchmark%/}
model=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      shift
      if [ "$#" -eq 0 ]; then
        echo "usage: sh train.sh [benchmark] --model MODEL" >&2
        exit 2
      fi
      model=$1
      shift
      ;;
    --model=*)
      model=${1#--model=}
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      echo "usage: sh train.sh [benchmark] --model MODEL" >&2
      exit 2
      ;;
  esac
done
if [ -z "${model}" ]; then
  echo "missing required --model MODEL" >&2
  echo "usage: sh train.sh [benchmark] --model MODEL" >&2
  exit 2
fi
tmp_run_dir="/tmp/zsheng1/${benchmark}"
run_dir="${benchmark}/run"

conda run -n checkpoint python "${benchmark}/finetune_benchmark.py" \
  --model "${model}" \
  --hook-types baseline async async_o gockpt gockpt_o \
  --seq-len 512 \
  --max-steps 207 \
  --save-steps 50 \
  --overlap-steps 2 \
  --gockpt-inflight-packets 64 \
  --gockpt-transfer-chunk-mb 64 \
  --gradient-checkpointing \
  --output-dir "${tmp_run_dir}"

rsync -avh --delete --progress --exclude='*.pt' "${tmp_run_dir}/" "${run_dir}/"

conda run -n checkpoint python "${benchmark}/visualize_checkpoint_power.py" \
    --report "${tmp_run_dir}/power.json" \
    --output "${benchmark}/images/checkpoint_power.png"

conda run -n checkpoint python "${benchmark}/visualize_checkpoint_save_time.py" \
    --report "${tmp_run_dir}/report.json" \
    --output "${benchmark}/images/foreground_checkpoint_time.png" \
    --title "Foreground Checkpoint Stall Time vs Algorithms"

conda run -n checkpoint python "${benchmark}/visualize_checkpoint_memory.py" \
    --report "${run_dir}/host_memory.json" \
    --output "${benchmark}/images/checkpoint_memory.png" \
    --title "Host Memory Usage Over Time"

conda run -n checkpoint python "${benchmark}/visualize_checkpoint_phase.py" \
    --report "${run_dir}/report.json" \
    -o "${benchmark}/images/checkpoint_phase.png"
