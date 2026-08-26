#!/usr/bin/env bash
# Itération rapide SANS rebuild : pousse le package python/sglang patché (branche
# soma/comfy-w4a8) dans le pod de dev, par-dessus la base lmsysorg/sglang:latest.
#
# Usage : ./sync-to-pod.sh [pod] [namespace]
#   défaut : pod=ltx-spike-sglang, ns=soma
#
# kubectl cp ne fait pas de --delete ; on tar/untar le sous-arbre pour rester simple
# et rapide (pas de rsync dans l'image sglang de base).
set -euo pipefail

POD="${1:-ltx-spike-sglang}"
NS="${2:-soma}"
SRC="$(cd "$(dirname "$0")" && pwd)/python/sglang"
DEST="/sgl-workspace/sglang/python/sglang"

echo "[sync] $SRC -> $NS/$POD:$DEST"
tar -C "$(dirname "$SRC")" -cf - sglang \
  | kubectl -n "$NS" exec -i "$POD" -- tar -C "$(dirname "$DEST")" -xf -
echo "[sync] done"
