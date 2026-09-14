#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
model_root="${NARRATION_MODEL_ROOT:-/data/huajiacheng/models/narration-family-pilot}"

declare -A expected_files=(
  [Qwen3-8B]=16
  [Qwen3-14B]=19
  [Hunyuan-4B-Instruct]=15
  [Hunyuan-7B-Instruct]=17
  [InternLM2.5-7B-Chat]=21
  [InternLM2.5-20B-Chat]=34
)
declare -A expected_bytes=(
  [Qwen3-8B]=16397461896
  [Qwen3-14B]=29552614406
  [Hunyuan-4B-Instruct]=8449905348
  [Hunyuan-7B-Instruct]=15025698663
  [InternLM2.5-7B-Chat]=15477069914
  [InternLM2.5-20B-Chat]=39723972560
)

mkdir -p "$project_dir/logs"
echo "[$(date --iso-8601=seconds)] waiting for six complete model copies"
while true; do
  ready=1
  for model in "${!expected_files[@]}"; do
    path="$model_root/$model"
    if [[ ! -d "$path" ]]; then
      ready=0
      continue
    fi
    files="$(find "$path" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' 2>/dev/null | wc -l)"
    bytes="$(find "$path" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s+0}')"
    if [[ "$files" -ne "${expected_files[$model]}" || "$bytes" -ne "${expected_bytes[$model]}" ]]; then
      ready=0
    fi
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 30
done

echo "[$(date --iso-8601=seconds)] models complete; starting pilot"
export NARRATION_MODEL_ROOT="$model_root"
export NARRATION_PYTHON="${NARRATION_PYTHON:-/home/huajiacheng/miniforge3/envs/vlm/bin/python}"
exec "$script_dir/launch_open_family_pilot20.sh"
