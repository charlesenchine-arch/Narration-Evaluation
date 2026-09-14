#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
cd "$project_dir"

run_dir="data/open_family_pilot20"
smoke_dir="data/open_family_pilot20_smoke_repair"
log_file="logs/open_family_pilot20.log"
models=(qwen3_8b qwen3_14b hunyuan_4b hunyuan_7b internlm2_5_7b internlm2_5_20b)

mkdir -p "$(dirname "$log_file")" "$smoke_dir"
exec >>"$log_file" 2>&1

export NARRATION_MODEL_ROOT="${NARRATION_MODEL_ROOT:-/nvme/jqhua/models}"
python_bin="${NARRATION_PYTHON:-.venv-pilot/bin/python}"
runner="scripts/run_open_family_pilot20.py"

wait_for_gpu() {
  local gpu still_free
  echo "[$(date -Is)] waiting for an entirely free GPU" >&2
  while true; do
    gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | \
      awk -F, '$2 + 0 < 1000 && $3 + 0 < 5 && first == "" {gsub(/ /, "", $1); first=$1} END {print first}')"
    if [[ -n "$gpu" ]]; then
      sleep 15
      still_free="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | \
        awk -F, -v target="$gpu" '$1 + 0 == target && $2 + 0 < 1000 && $3 + 0 < 5 {print "yes"}')"
      if [[ "$still_free" == "yes" ]]; then
        echo "$gpu"
        return 0
      fi
    fi
    sleep 30
  done
}

run_on_free_gpu() {
  local label="$1"
  shift
  local gpu
  gpu="$(wait_for_gpu)"
  echo "[$(date -Is)] using physical GPU $gpu for $label"
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

echo "[$(date -Is)] starting/resuming family smoke tests"
for model in qwen3_8b hunyuan_4b internlm2_5_7b; do
  run_on_free_gpu "smoke generation $model" \
    "$python_bin" "$runner" --work-dir "$smoke_dir" --limit 1 generate --model "$model"
  MODEL="$model" SMOKE_DIR="$smoke_dir" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SMOKE_DIR"]) / "generations" / f"{os.environ['MODEL']}.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not any(row.get("prompt_id") == "P01" and row.get("status") == "ok" for row in rows):
    raise SystemExit(f"smoke QC failed: {path}")
PY
done

echo "[$(date -Is)] smoke tests passed; starting complete generation"
prepared=0
for generation_round in 1 2 3 4 5 6; do
  echo "[$(date -Is)] generation QC round $generation_round"
  for model in "${models[@]}"; do
    run_on_free_gpu "generation $model" \
      "$python_bin" "$runner" --work-dir "$run_dir" generate --model "$model"
  done
  if "$python_bin" "$runner" --work-dir "$run_dir" prepare; then
    prepared=1
    break
  fi
done
if [[ "$prepared" -ne 1 ]]; then
  echo "[$(date -Is)] generation QC did not converge after six rounds"
  exit 1
fi

echo "[$(date -Is)] generation passed QC; starting blind AB/BA judging"
for model in "${models[@]}"; do
  run_on_free_gpu "judging $model" \
    "$python_bin" "$runner" --work-dir "$run_dir" judge --model "$model"
done

"$python_bin" "$runner" --work-dir "$run_dir" analyze
date -Is >"$run_dir/COMPLETE"
echo "[$(date -Is)] complete"
