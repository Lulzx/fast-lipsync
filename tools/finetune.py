"""Person-specific LoRA fine-tune of the MuseTalk 1.5 UNet on the speaker's own footage.

Each sample: the speaker's face crop with the lower half masked + a reference crop from within REF_WINDOW frames
+ the Whisper features of that moment; target = the real face. Losses: latent L1, plus pixel L1 and VGG
perceptual loss on the lower half after decoding with TAESD (the decoder lip-sync uses). Only LoRA weights train,
which keeps memory well inside a 24 GB Mac.

Usage: python tools/finetune.py footage.mp4 [--steps 3000]
Writes models/musetalk_ft/<source stem>/unet.pth (LoRA merged into the full UNet).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lipsync as L  # noqa: E402

from diffusers import AutoencoderTiny, UNet2DConditionModel  # noqa: E402
from musetalk.models.unet import PositionalEncoding  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torchvision.models import VGG16_Weights, vgg16  # noqa: E402

DEV = L.DEVICE
BATCH = 8
LR = 1e-4
RANK = 32
REF_WINDOW = 125        # reference frame within +-5 s, like run-time references from the same scene
W_LATENT, W_PIXEL, W_VGG = 1.0, 1.0, 0.5
TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2", "ff.net.0.proj", "ff.net.2"]


class VGGLoss(torch.nn.Module):
    """L1 between VGG16 relu1_2 / relu2_2 / relu3_3 features (ImageNet weights, frozen)."""

    def __init__(self):
        super().__init__()
        f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in f.parameters():
            p.requires_grad_(False)
        self.slices = torch.nn.ModuleList([f[:4], f[4:9], f[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):  # x, y in [-1, 1], RGB
        x, y = ((x + 1) / 2 - self.mean) / self.std, ((y + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + (x - y).abs().mean()
        return loss


class Data:
    def __init__(self, d):
        self.frames = np.load(d / "frames.npy")
        self.crops = np.load(d / "crops.npy", mmap_mode="r")  # BGR uint8 256x256
        self.masked = torch.from_numpy(np.load(d / "masked.npy")).float()
        self.full = torch.from_numpy(np.load(d / "full.npy")).float()
        self.audio = torch.from_numpy(np.load(d / "audio.npy")).float()
        val = np.load(d / "val.npy")
        self.train_idx, self.val_idx = np.flatnonzero(~val), np.flatnonzero(val)
        self.rng = np.random.default_rng(0)

    def ref_for(self, i):
        lo, hi = np.searchsorted(self.frames, [self.frames[i] - REF_WINDOW, self.frames[i] + REF_WINDOW])
        return int(self.rng.integers(lo, max(hi, lo + 1)))

    def batch(self, idx):
        refs = [self.ref_for(i) for i in idx]
        img = torch.from_numpy(np.stack([self.crops[i][..., ::-1] for i in idx]).copy()).permute(0, 3, 1, 2)
        return (torch.cat([self.masked[idx], self.full[refs]], 1), self.audio[idx], self.full[idx],
                img.float() / 127.5 - 1)


def build_unet():
    mt = L.MT / "models/musetalkV15"
    unet = UNet2DConditionModel(**json.loads((mt / "musetalk.json").read_text()))
    unet.load_state_dict(torch.load(mt / "unet.pth", map_location="cpu"))
    unet.requires_grad_(False)
    return get_peft_model(unet, LoraConfig(r=RANK, lora_alpha=RANK, target_modules=TARGETS, lora_dropout=0.0))


def step_loss(unet, pe, tae, vgg, batch, t):
    lat, aud, target, img = (x.to(DEV) for x in batch)
    pred = unet(lat, t, encoder_hidden_states=pe(aud)).sample
    loss_lat = F.l1_loss(pred, target)
    dec = tae.decode(pred).sample                     # [-1, 1]
    lower = slice(128, 256)                           # the generated half of the crop
    loss_pix = F.l1_loss(dec[..., lower, :], img[..., lower, :])
    loss_vgg = vgg(dec[..., lower, :], img[..., lower, :])
    return W_LATENT * loss_lat + W_PIXEL * loss_pix + W_VGG * loss_vgg, (loss_lat.item(), loss_pix.item(), loss_vgg.item())


def main(src, steps):
    stem = Path(src).stem
    data = Data(L.ROOT / "cache/train" / stem)
    out = L.ROOT / "models/musetalk_ft" / stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(data.train_idx)} train / {len(data.val_idx)} val samples")

    unet = build_unet().to(DEV)
    unet.print_trainable_parameters()
    pe = PositionalEncoding(d_model=384).to(DEV)
    tae = AutoencoderTiny.from_pretrained(str(L.ROOT / "models/taesd")).to(DEV).eval().requires_grad_(False)
    vgg = VGGLoss().to(DEV)
    t = torch.tensor([0], device=DEV)
    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    val_idx = data.val_idx[np.linspace(0, len(data.val_idx) - 1, 64).astype(int)]

    def validate():
        unet.eval()
        tot = np.zeros(3)
        with torch.no_grad():
            for k in range(0, len(val_idx), BATCH):
                tot += step_loss(unet, pe, tae, vgg, data.batch(val_idx[k:k + BATCH]), t)[1]
        unet.train()
        return tot / (len(val_idx) // BATCH)

    print("val before:", np.round(validate(), 4), flush=True)
    t0, run = time.time(), np.zeros(3)
    for step in range(1, steps + 1):
        idx = data.rng.choice(data.train_idx, BATCH, replace=False)
        loss, parts = step_loss(unet, pe, tae, vgg, data.batch(idx), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += parts
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps}  lat/pix/vgg {np.round(run / 50, 4)}  {el / step:.2f}s/step  "
                  f"eta {el / step * (steps - step) / 60:.0f} min", flush=True)
            run[:] = 0
        if step % 500 == 0 or step == steps:
            print(f"val @ {step}:", np.round(validate(), 4), flush=True)
            unet.save_pretrained(out / "lora")
    merged = unet.merge_and_unload()
    torch.save(merged.state_dict(), out / "unet.pth")
    print(f"done in {(time.time() - t0) / 60:.0f} min -> {out / 'unet.pth'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    main(a.source, a.steps)
