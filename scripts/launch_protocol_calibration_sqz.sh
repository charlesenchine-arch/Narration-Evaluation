#!/usr/bin/env bash
set -euo pipefail

project_dir="${NARRATION_PROJECT_DIR:-/home/huajiacheng/codes/Narration-Evaluation}"
model_root="${NARRATION_MODEL_ROOT:-/data/huajiacheng/models/narration-family-pilot}"
python_bin="${NARRATION_PYTHON:-/home/huajiacheng/miniforge3/envs/vlm/bin/python}"
work_dir="$project_dir/data/open_family_pilot20"
out_dir="$work_dir/protocol_calibration"

cd "$project_dir"
export NARRATION_MODEL_ROOT="$model_root"
export CUDA_VISIBLE_DEVICES="${NARRATION_CALIBRATION_GPU:-4}"

for model in qwen3_8b qwen3_14b hunyuan_4b internlm2_5_7b hunyuan_7b internlm2_5_20b; do
  echo "[$(date --iso-8601=seconds)] protocol calibration: $model"
  if [[ "$model" == "internlm2_5_20b" ]]; then
    NARRATION_GPU_MEMORY_UTILIZATION=0.99 NARRATION_ENFORCE_EAGER=1 \
      "$python_bin" scripts/run_judge_protocol_calibration.py \
      --work-dir "$work_dir" --out-dir "$out_dir" run --model "$model"
  else
    "$python_bin" scripts/run_judge_protocol_calibration.py \
      --work-dir "$work_dir" --out-dir "$out_dir" run --model "$model"
  fi
done

"$python_bin" scripts/run_judge_protocol_calibration.py \
  --work-dir "$work_dir" --out-dir "$out_dir" analyze
date --iso-8601=seconds > "$out_dir/PROTOCOL_CALIBRATION_COMPLETE"
