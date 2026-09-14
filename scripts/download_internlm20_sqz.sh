#!/usr/bin/env bash
set -euo pipefail

model_root="${NARRATION_MODEL_ROOT:-/data/huajiacheng/models/narration-family-pilot}"
temporary_root="${NARRATION_DIRECT_ROOT:-/data/huajiacheng/models/narration-family-pilot-direct}"
model_name="InternLM2.5-20B-Chat"
temporary_dir="$temporary_root/$model_name"
final_dir="$model_root/$model_name"
modelscope_bin="${MODELSCOPE_BIN:-/home/huajiacheng/miniforge3/envs/vlm/bin/modelscope}"

# Reuse the direct-download workers after Qwen3-14B finishes instead of making
# both eight-worker downloads compete with each other.
while [[ ! -f "$temporary_root/Qwen3-14B.COMPLETE" ]]; do
  sleep 30
done

mkdir -p "$temporary_dir" "$final_dir"
"$modelscope_bin" download \
  --model "Shanghai_AI_Laboratory/internlm2_5-20b-chat" \
  --local_dir "$temporary_dir" \
  --max-workers 8

files="$(find "$temporary_dir" -type f ! -name '.msc' ! -name '.mv' ! -path "$temporary_dir/.cache/*" ! -path "$temporary_dir/._____temp/*" | wc -l)"
bytes="$(find "$temporary_dir" -type f ! -name '.msc' ! -name '.mv' ! -path "$temporary_dir/.cache/*" ! -path "$temporary_dir/._____temp/*" -printf '%s\n' | awk '{s+=$1} END {printf "%.0f", s+0}')"
if [[ "$files" -ne 33 || "$bytes" -ne 39723971041 ]]; then
  echo "direct InternLM2.5-20B validation failed before promotion: files=$files bytes=$bytes" >&2
  exit 1
fi

mapfile -t competing < <(pgrep -f "rsync .*InternLM2.5-20B-Chat.*narration-family-pilot/InternLM2.5-20B-Chat" || true)
if [[ "${#competing[@]}" -gt 0 ]]; then
  kill "${competing[@]}" || true
  sleep 2
fi

rsync -a --exclude='.cache/' --exclude='._____temp/' --exclude='.msc' --exclude='.mv' "$temporary_dir/" "$final_dir/"
date --iso-8601=seconds > "$temporary_root/InternLM2.5-20B-Chat.COMPLETE"
echo "InternLM2.5-20B direct download promoted successfully"
