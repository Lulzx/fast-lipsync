"""Convert the MuseTalk 1.5 UNet to Core ML (fp16, fixed batch) so it can run on the Neural Engine + GPU.

Usage: python tools/convert_unet_coreml.py [batch] [--weights models/musetalk_ft/<stem>/unet.pth --name <stem>]
"""
import json
import sys
from pathlib import Path

import coremltools as ct
import torch
from diffusers import UNet2DConditionModel

ROOT = Path(__file__).resolve().parent.parent
MT = ROOT / "vendor/MuseTalk/models/musetalkV15"


class Wrapped(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.register_buffer("t", torch.tensor([0]))

    def forward(self, latents, audio):
        return self.unet(latents, self.t, encoder_hidden_states=audio, return_dict=False)[0]


def main(batch=8, weights=None, name=None):
    unet = UNet2DConditionModel(**json.loads((MT / "musetalk.json").read_text()))
    unet.load_state_dict(torch.load(weights or MT / "unet.pth", map_location="cpu"))
    m = Wrapped(unet.eval()).eval()
    lat, aud = torch.randn(batch, 8, 32, 32), torch.randn(batch, 50, 384)
    with torch.no_grad():
        traced = torch.jit.trace(m, (lat, aud))
    ml = ct.convert(traced, inputs=[ct.TensorType("latents", lat.shape), ct.TensorType("audio", aud.shape)],
                    outputs=[ct.TensorType("pred")], compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
    out = ROOT / f"models/coreml/musetalk_unet{'_' + name if name else ''}_b{batch}.mlpackage"
    ml.save(str(out))
    print(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int, nargs="?", default=8)
    ap.add_argument("--weights")
    ap.add_argument("--name", help="suffix for the package, e.g. the source stem of a fine-tuned model")
    a = ap.parse_args()
    main(a.batch, a.weights, a.name)
