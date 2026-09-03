#!/usr/bin/env bash
# check_env.sh — print the 5 coordinates that determine which prebuilt flash-attn
# wheel (or any torch-linked CUDA wheel) will install + import on this machine, then
# query the flash-attn release API for wheels that match THIS box.
#
# Usage:
#   ./check_env.sh                  # uses `python` on PATH
#   PY=.venv/bin/python ./check_env.sh

set -euo pipefail

PY="${PY:-python}"
REPO="${FA_REPO:-Dao-AILab/flash-attention}"

echo "== environment coordinates =="
read -r TORCH CUDA ABI <<<"$("$PY" - <<'PYEOF'
import torch
cuda = torch.version.cuda or "none"
abi  = torch._C._GLIBCXX_USE_CXX11_ABI
print(torch.__version__.split('+')[0], cuda, "TRUE" if abi else "FALSE")
PYEOF
)"

PYTAG="cp$("$PY" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
ARCH="$(uname -m)"
OSID="$(uname -s | tr '[:upper:]' '[:lower:]')"

CUMAJ="cu${CUDA%%.*}"
TORCH_MM="$(echo "$TORCH" | cut -d. -f1,2)"

printf '  torch        : %s  (wheel tag: torch%s)\n' "$TORCH" "$TORCH_MM"
printf '  CUDA         : %s  (wheel tag: %s)\n'       "$CUDA"  "$CUMAJ"
printf '  C++ ABI      : %s  (wheel tag: cxx11abi%s)\n' "$ABI" "$ABI"
printf '  python       : %s\n'                         "$PYTAG"
printf '  OS / arch    : %s / %s  (wheel tag: %s_%s)\n' "$OSID" "$ARCH" "$OSID" "$ARCH"

if [[ "$CUDA" == "none" ]]; then
  echo
  echo "!! torch reports no CUDA — this is a CPU-only build; flash-attn won't apply here."
  exit 0
fi

echo
echo "== matching flash-attn wheels in latest release of $REPO =="
echo "   (filter: ${CUMAJ} / torch${TORCH_MM} / cxx11abi${ABI} / ${PYTAG} / ${ARCH})"
echo

MATCH="${CUMAJ}torch${TORCH_MM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-${OSID}_${ARCH}"

URLS="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
  | grep -oE '"browser_download_url": *"[^"]+\.whl"' \
  | sed 's/.*: *"//; s/"$//' \
  | grep -F "$MATCH" || true)"

if [[ -z "$URLS" ]]; then
  echo "  (none found for these exact coordinates)"
  echo
  echo "  Tip: if your torch version has no matching wheel, find the newest torch that"
  echo "  DOES have one for your ${CUMAJ}/${ABI}/${PYTAG}, then pin torch to that."
  echo "  Browse all assets:"
  echo "    curl -fsSL https://api.github.com/repos/${REPO}/releases/latest \\"
  echo "      | grep -oE 'https://[^\"]+\\.whl' | grep ${PYTAG}"
else
  echo "$URLS" | while read -r u; do echo "  $u"; done
  echo
  echo "  Install with:"
  echo "    $PY -m pip install \"$(echo "$URLS" | head -1)\""
fi
