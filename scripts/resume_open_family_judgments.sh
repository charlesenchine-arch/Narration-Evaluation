#!/usr/bin/env bash
set -euo pipefail

cd /nvme/jqhua/Narration-Evaluation
export PATH="/nvme/jqhua/miniconda3/envs/vlm/bin:$PATH"

run_dir="data/open_family_pilot20"
log_file="logs/open_family_pilot20.log"
python_bin=".venv-pilot/bin/python"
runner="scripts/run_open_family_pilot20.py"

exec >>"$log_file" 2>&1
rm -f "$run_dir/COMPLETE"

wait_for_gpu() {
  local gpu still_free
  echo "[$(date -Is)] waiting for an entirely free GPU for judgment repair"
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

for model in qwen3_8b hunyuan_7b; do
  gpu="$(wait_for_gpu | tail -n 1)"
  echo "[$(date -Is)] repairing judgments for $model on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$runner" --work-dir "$run_dir" judge --model "$model"
done

RUN_DIR="$run_dir" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

models = ["qwen3_8b", "qwen3_14b", "hunyuan_4b", "hunyuan_7b", "internlm2_5_7b", "internlm2_5_20b"]
root = Path(os.environ["RUN_DIR"]) / "judgments"
for model in models:
    rows = [json.loads(line) for line in (root / f"{model}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {row["judgment_id"] for row in rows if row.get("status") == "ok"}
    if len(completed) != 600:
        raise SystemExit(f"incomplete judgments for {model}: {len(completed)}/600")
PY

"$python_bin" "$runner" --work-dir "$run_dir" analyze
date -Is >"$run_dir/COMPLETE"
echo "[$(date -Is)] judgment repair complete"
