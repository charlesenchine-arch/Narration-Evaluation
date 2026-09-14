#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
model_root="${NARRATION_MODEL_ROOT:-/data/huajiacheng/models/narration-family-pilot}"
python_bin="${NARRATION_PYTHON:-/home/huajiacheng/miniforge3/envs/vlm/bin/python}"
work_dir="$project_dir/data/open_family_pilot20"
audit_dir="$work_dir/audit"

declare -A model_dirs=(
  [qwen3_8b]=Qwen3-8B
  [qwen3_14b]=Qwen3-14B
  [hunyuan_4b]=Hunyuan-4B-Instruct
  [hunyuan_7b]=Hunyuan-7B-Instruct
  [internlm2_5_7b]=InternLM2.5-7B-Chat
  [internlm2_5_20b]=InternLM2.5-20B-Chat
)
declare -A expected_files=(
  [qwen3_8b]=16
  [qwen3_14b]=19
  [hunyuan_4b]=15
  [hunyuan_7b]=17
  [internlm2_5_7b]=21
  [internlm2_5_20b]=34
)
declare -A expected_bytes=(
  [qwen3_8b]=16397461896
  [qwen3_14b]=29552614406
  [hunyuan_4b]=8449905348
  [hunyuan_7b]=15025698663
  [internlm2_5_7b]=15477069914
  [internlm2_5_20b]=39723972560
)

model_ready() {
  local key="$1" path files bytes
  path="$model_root/${model_dirs[$key]}"
  [[ -d "$path" ]] || return 1
  files="$(find "$path" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' | wc -l)"
  bytes="$(find "$path" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s+0}')"
  [[ "$files" -eq "${expected_files[$key]}" && "$bytes" -eq "${expected_bytes[$key]}" ]]
}

cd "$project_dir"
export NARRATION_MODEL_ROOT="$model_root"
export CUDA_VISIBLE_DEVICES="${NARRATION_AUDIT_GPU:-4}"
for model in qwen3_8b qwen3_14b hunyuan_4b internlm2_5_7b hunyuan_7b internlm2_5_20b; do
  while ! model_ready "$model"; do
    echo "[$(date --iso-8601=seconds)] waiting for complete $model"
    sleep 30
  done
  echo "[$(date --iso-8601=seconds)] same-order repeats: $model"
  "$python_bin" scripts/run_same_order_repeat_audit.py \
    --work-dir "$work_dir" --audit-dir "$audit_dir" run --model "$model"
done
"$python_bin" scripts/run_same_order_repeat_audit.py \
  --work-dir "$work_dir" --audit-dir "$audit_dir" analyze
date --iso-8601=seconds > "$audit_dir/SAME_ORDER_COMPLETE"
echo "[$(date --iso-8601=seconds)] same-order repeat audit complete"
