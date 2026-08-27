#!/bin/bash
# Disaggregated LTX-2 W4A8 in ONE container, sequential roles, .bin hand-off.
# encoder (Gemma only) -> embeds.bin -> denoiser (DiT only) -> latents.bin ->
# decoder (VAE only) -> mp4. Each `sglang generate` is a separate process that
# dies before the next, so only ONE component is resident at a time (RAM+VRAM
# minimized -> 1080p unblocked, no cgroup thrash). sglang perf kept per stage.
set -euo pipefail

MODEL="${MODEL:-/data/spike/ltx-2.5-local}"
W4A8="${W4A8:-/data/ltx/w4a8/ltx-2.5-22b-distilled-transformer_W4A8_Mixed.safetensors}"
PROMPT="${PROMPT:-A single colossal bioluminescent jellyfish drifts through a deep ocean trench, slow steady push.}"
WIDTH="${WIDTH:-832}"; HEIGHT="${HEIGHT:-480}"; FRAMES="${FRAMES:-49}"
STEPS="${STEPS:-8}"; SEED="${SEED:-42}"
OUT="${OUT:-/data/spike/out/optimized}"
TMP="${TMP:-/data/spike/out/tmp_$$}"
mkdir -p "$TMP" "$(dirname "$OUT")"
trap 'rm -f "$TMP"/embeds.bin "$TMP"/latents.bin; rmdir "$TMP" 2>/dev/null || true' EXIT

COMMON=(
  --model-path="$MODEL" --model-id=LTX-2.5-Diffusers
  --transformer-weights-path="$W4A8"
  --prompt="$PROMPT" --width="$WIDTH" --height="$HEIGHT" --num-frames="$FRAMES"
  --num-inference-steps="$STEPS" --seed="$SEED" --num-gpus=1
  --cpu-offload-components connectors vae vocoder audio_vae
  --dit-layerwise-offload --dit-offload-prefetch-size 0
)

echo "=================== [1/3] ENCODER (Gemma only) ==================="
SOMA_DUMP_PAYLOAD="$TMP/embeds.bin" \
  sglang generate --disagg-role encoder "${COMMON[@]}" \
  --layerwise-offload-components text_encoder

echo "=================== [2/3] DENOISER (DiT only) ==================="
SOMA_LOAD_PAYLOAD="$TMP/embeds.bin" SOMA_DUMP_PAYLOAD="$TMP/latents.bin" \
  sglang generate --disagg-role denoiser "${COMMON[@]}"

echo "=================== [3/3] DECODER (VAE only) -> mp4 ==================="
SOMA_LOAD_PAYLOAD="$TMP/latents.bin" \
  sglang generate --disagg-role decoder "${COMMON[@]}" \
  --output-file-path="$OUT" --save-output

echo "=================== DONE: ${OUT}.mp4 ==================="
