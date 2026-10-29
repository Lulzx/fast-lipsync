#!/bin/zsh
# One-time setup on an Apple Silicon Mac: Python env, MuseTalk (patched for MPS), model weights, Core ML UNet.
set -euo pipefail
cd "${0:A:h}"

uv venv -q -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt
HF=.venv/bin/huggingface-cli

[[ -d vendor/MuseTalk ]] || git clone --depth 1 https://github.com/TMElyralab/MuseTalk vendor/MuseTalk
(cd vendor/MuseTalk && git apply --check ../../patches/musetalk-mps.patch 2>/dev/null && git apply ../../patches/musetalk-mps.patch || true)
M=vendor/MuseTalk/models
$HF download TMElyralab/MuseTalk --local-dir $M --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
$HF download stabilityai/sd-vae-ft-mse --local-dir $M/sd-vae --include "config.json" "diffusion_pytorch_model.bin"
$HF download openai/whisper-tiny --local-dir $M/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

mkdir -p models/taesd models/coreml models/insightface/models/buffalo_l
$HF download madebyollin/taesd --local-dir models/taesd --include "config.json" "diffusion_pytorch_model.safetensors"
[[ -f models/hand_landmarker.task ]] || curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
if [[ ! -f models/insightface/models/buffalo_l/det_10g.onnx ]]; then
  tmp=$(mktemp -d)
  curl -sSL -o $tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  unzip -o -j $tmp/buffalo_l.zip det_10g.onnx 2d106det.onnx -d models/insightface/models/buffalo_l
  rm -rf $tmp
fi
[[ -d models/coreml/musetalk_unet_b8.mlpackage ]] || .venv/bin/python -I tools/convert_unet_coreml.py 8
echo "Done."
