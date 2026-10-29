#!/bin/zsh
# Lip-sync footage to a voice track: ./lipsync.sh footage.mp4 voice.wav out.mp4 [--start 12.5] [--mouth real] ...
set -euo pipefail
ROOT=${0:A:h}
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec $ROOT/.venv/bin/python -I $ROOT/tools/lipsync.py "$@"
