#!/usr/bin/env bash
set -euo pipefail

model_root="${NARRATION_MODEL_ROOT:-/data/huajiacheng/models/narration-family-pilot}"
temporary_root="${NARRATION_DIRECT_ROOT:-/data/huajiacheng/models/narration-family-pilot-direct}"
model_name="Qwen3-14B"
temporary_dir="$temporary_root/$model_name"
final_dir="$model_root/$model_name"
modelscope_bin="${MODELSCOPE_BIN:-/home/huajiacheng/miniforge3/envs/vlm/bin/modelscope}"

mkdir -p "$temporary_dir" "$final_dir"
"$modelscope_bin" download \
  --model "Qwen/$model_name" \
  --local_dir "$temporary_dir" \
  --max-workers 8

# If the slower SY copy has reached this model, stop only that model's rsync
# before promoting the verified independent download.
mapfile -t competing < <(pgrep -f "rsync .*Qwen3-14B.*narration-family-pilot/Qwen3-14B" || true)
if [[ "${#competing[@]}" -gt 0 ]]; then
  kill "${competing[@]}" || true
  sleep 2
fi

rsync -a --exclude='.cache/' --exclude='._____temp/' --exclude='.msc' --exclude='.mv' "$temporary_dir/" "$final_dir/"
files="$(find "$final_dir" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' | wc -l)"
bytes="$(find "$final_dir" -type f ! -name '.msc' ! -name '.mv' ! -path '*/.cache/*' ! -path '*/._____temp/*' -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s+0}')"
if [[ "$files" -ne 19 || "$bytes" -ne 29552614406 ]]; then
  echo "direct Qwen3-14B validation failed: files=$files bytes=$bytes" >&2
  exit 1
fi
date --iso-8601=seconds > "$temporary_root/Qwen3-14B.COMPLETE"
echo "Qwen3-14B direct download promoted successfully"
