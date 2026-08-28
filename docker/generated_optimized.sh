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
REPORTS="${REPORTS:-$(dirname "$OUT")/perf_$$}"
# The perf report has no generation-time cost (a separate python process on the
# tiny JSONs, after the run); SOMA_REPORT=0 skips it anyway. It ships inside the
# package (rides the existing fork mount) and is invoked with `python3 -m`, so
# no extra mount is needed. SOMA_PERF_REPORT=<file> forces a specific script.
SOMA_REPORT="${SOMA_REPORT:-1}"
# The aggregator lives in the fork tree (rides the existing python mount) and is
# stdlib-only, so it is run BY PATH — not `python3 -m`, which would import the
# whole sglang/CUDA package just to print a table. SOMA_PERF_REPORT overrides.
REPORT_PY="${SOMA_PERF_REPORT:-/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/disaggregation/soma_perf_report.py}"
# In a container $$ is always 1, so perf_$$ is reused across runs — a prior
# run's per-role JSONs would linger and get mixed in. Start from a clean dir.
rm -rf "$REPORTS"
mkdir -p "$TMP" "$REPORTS" "$(dirname "$OUT")"
trap 'rm -f "$TMP"/embeds.bin "$TMP"/latents.bin; rmdir "$TMP" 2>/dev/null || true' EXIT

COMMON=(
  --model-path="$MODEL" --model-id=LTX-2.5-Diffusers
  --transformer-weights-path="$W4A8"
  --prompt="$PROMPT" --width="$WIDTH" --height="$HEIGHT" --num-frames="$FRAMES"
  --num-inference-steps="$STEPS" --seed="$SEED" --num-gpus=1
  --cpu-offload-components connectors vae vocoder audio_vae
)

# DIT_OFFLOAD=1 (défaut) : layerwise offload du DiT (stream, VRAM mini, mais RAM
#   CPU pinned/mapped → sujet au bug cgroup). DIT_OFFLOAD=0 : DiT RÉSIDENT en VRAM
#   (rentre en disagg car le denoiser est seul ; pas de RAM CPU → esquive le bug
#   cgroup, et plus rapide car zéro H2D). N'affecte que le denoiser (seul à charger
#   le transformer ; encoder/decoder le skippent).
DIT_OFFLOAD="${DIT_OFFLOAD:-1}"
if [ "$DIT_OFFLOAD" = "1" ]; then
  COMMON+=(--dit-layerwise-offload --dit-offload-prefetch-size 0)
fi

# SageAttention sur le DENOISER (attention quantifiée/fusionnée). MESURÉ ~36%
# plus rapide à 1920x1088x121 (15424 -> 9803 ms/step), qualité préservée ; le
# gain croît avec la durée (attention O(N²)). Le paquet compilé (2.2.0, cu130/
# sm89) vit sur le PVC en /data/sage-pkgs → exposé via PYTHONPATH. Sans lui,
# sglang retombe silencieusement sur Flash Attention. SAGE=0 pour couper.
SAGE="${SAGE:-1}"
SAGE_PKGS="${SAGE_PKGS:-/data/sage-pkgs}"
SAGE_ARGS=()
if [ "$SAGE" = "1" ] && [ -d "$SAGE_PKGS" ]; then
  export PYTHONPATH="$SAGE_PKGS${PYTHONPATH:+:$PYTHONPATH}"
  SAGE_ARGS=(--attention-backend sage_attn)
  echo "SageAttention ON (denoiser) — PYTHONPATH=$SAGE_PKGS"
elif [ "$SAGE" = "1" ]; then
  echo "SAGE=1 mais $SAGE_PKGS absent → Flash Attention (installer SageAttention sur le PVC)"
fi

# Per-role extra flags, appended AFTER COMMON so they override it (argparse:
# last wins). This is the tuning lever the report exposes — e.g. keep DiT layers
# resident instead of re-streaming all 48 every step:
#   EXTRA_DENOISER="--dit-layerwise-resident-layers 40 --dit-offload-prefetch-size 2"
# Word-split on purpose (these are CLI flags, no globs).
EXTRA_ENCODER="${EXTRA_ENCODER:-}"
EXTRA_DENOISER="${EXTRA_DENOISER:-}"
EXTRA_DECODER="${EXTRA_DECODER:-}"

# wall-clock (load + inference) per role; the JSON carries pure inference time,
# so wall - infer = the per-role model-load cost. `SECONDS` resets before each.
echo "=================== [1/3] ENCODER (Gemma only) ==================="
SECONDS=0
SOMA_DUMP_PAYLOAD="$TMP/embeds.bin" \
  sglang generate --disagg-role encoder "${COMMON[@]}" \
  --layerwise-offload-components text_encoder \
  --perf-dump-path "$REPORTS/encoder.json" $EXTRA_ENCODER
ENC_WALL=$SECONDS

echo "=================== [2/3] DENOISER (DiT only) ==================="
SECONDS=0
SOMA_LOAD_PAYLOAD="$TMP/embeds.bin" SOMA_DUMP_PAYLOAD="$TMP/latents.bin" \
  sglang generate --disagg-role denoiser "${COMMON[@]}" "${SAGE_ARGS[@]}" \
  --perf-dump-path "$REPORTS/denoiser.json" $EXTRA_DENOISER
DEN_WALL=$SECONDS

echo "=================== [3/3] DECODER (VAE only) -> mp4 ==================="
SECONDS=0
SOMA_LOAD_PAYLOAD="$TMP/latents.bin" \
  sglang generate --disagg-role decoder "${COMMON[@]}" \
  --output-file-path="$OUT" --save-output \
  --perf-dump-path "$REPORTS/decoder.json" $EXTRA_DECODER
DEC_WALL=$SECONDS

echo "=================== DONE: ${OUT}.mp4 ==================="
# The mp4 is the deliverable; never let the report step fail the run (|| true).
ARGS=(
  "${OUT}.report.json"
  "encoder:$REPORTS/encoder.json:$ENC_WALL"
  "denoiser:$REPORTS/denoiser.json:$DEN_WALL"
  "decoder:$REPORTS/decoder.json:$DEC_WALL"
)
if [ "$SOMA_REPORT" != "1" ]; then
  echo "perf report disabled (SOMA_REPORT=$SOMA_REPORT). JSONs in $REPORTS"
elif [ -f "$REPORT_PY" ]; then
  python3 "$REPORT_PY" "${ARGS[@]}" || true
else
  echo "perf report skipped: $REPORT_PY introuvable (set SOMA_PERF_REPORT). JSONs in $REPORTS"
fi
