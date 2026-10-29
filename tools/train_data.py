"""Build a person-specific MuseTalk fine-tuning set from the speaker's raw footage.

Frames come from the mouth library (front-facing, no hand near the mouth). For each: the MuseTalk face crop
(256x256), SD-VAE latents of the masked and full crop, and the Whisper audio feature for that frame. The last
10% of the footage is held out for validation.

Usage: python tools/train_data.py footage.mp4 [--max 8000]
Writes cache/train/<source stem>/{frames, crops, masked, full, audio, val}.npy
"""
import argparse
import subprocess
import time
from pathlib import Path

import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lipsync as L  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402


def usable_frames(stem):
    d = L.ROOT / "cache/mouths" / stem
    frames, desc, ok = np.load(d / "frames.npy"), np.load(d / "desc.npy"), np.load(d / "ok.npy")
    pose = desc[:, -4:]
    dev = np.abs(pose - np.median(pose, 0))
    return frames[ok & (dev < 3 * (np.median(dev, 0) + 1e-6)).all(1)]


@torch.no_grad()
def main(src, max_frames):
    src = Path(src)
    out = L.ROOT / "cache/train" / src.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = cv2.VideoCapture(str(src))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert round(fps) == L.FPS, f"training expects {L.FPS} fps footage, got {fps}"

    frames = usable_frames(src.stem)
    if len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    print(f"{len(frames)} frames from {total}")

    # audio features for every video frame, then keep the chosen ones
    wav = out / "audio.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    from transformers import WhisperModel
    whisper = WhisperModel.from_pretrained(str(L.MT / "models/whisper")).to(L.DEVICE, L.DTYPE).eval()
    ap = AudioProcessor(feature_extractor_path=str(L.MT / "models/whisper"))
    feats, length = ap.get_audio_feature(str(wav), weight_dtype=L.DTYPE)
    chunks = ap.get_whisper_chunk(feats, L.DEVICE, L.DTYPE, whisper, length, fps=L.FPS,
                                  audio_padding_length_left=2, audio_padding_length_right=2)
    frames = frames[frames < len(chunks)]
    np.save(out / "audio.npy", torch.stack([chunks[i] for i in frames]).cpu().numpy().astype(np.float16))
    del whisper
    print(f"audio features done ({time.time() - t0:.0f}s)")

    face, vae = L.load_face_detector(), L.load_vae()
    keep, crops, masked, full = [], [], [], []
    batch, want = [], set(frames.tolist())
    for i in range(total):
        if not cap.grab():
            break
        if i not in want:
            continue
        _, f = cap.retrieve()
        d = L.detect_face(f, face)
        if d is None:
            continue
        (box,), _, _ = L.smooth_faces([d], f.shape[0], f.shape[1])
        x1, y1, x2, y2 = box
        crop = cv2.resize(f[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        keep.append(i)
        crops.append(crop)
        batch.append(crop)
        if len(batch) == 16 or i == frames[-1]:
            m = vae.encode_latents(torch.cat([vae.preprocess_img(c, half_mask=True) for c in batch]).to(L.DEVICE))
            r = vae.encode_latents(torch.cat([vae.preprocess_img(c, half_mask=False) for c in batch]).to(L.DEVICE))
            masked.append(m.cpu().numpy().astype(np.float16))
            full.append(r.cpu().numpy().astype(np.float16))
            batch = []
        if len(keep) % 1000 == 0:
            print(f"  {len(keep)} crops ({time.time() - t0:.0f}s)", flush=True)
    if batch:
        m = vae.encode_latents(torch.cat([vae.preprocess_img(c, half_mask=True) for c in batch]).to(L.DEVICE))
        r = vae.encode_latents(torch.cat([vae.preprocess_img(c, half_mask=False) for c in batch]).to(L.DEVICE))
        masked.append(m.cpu().numpy().astype(np.float16))
        full.append(r.cpu().numpy().astype(np.float16))

    keep = np.array(keep)
    audio = np.load(out / "audio.npy")[np.searchsorted(frames, keep)]
    np.save(out / "audio.npy", audio)
    np.save(out / "frames.npy", keep)
    np.save(out / "crops.npy", np.stack(crops))
    np.save(out / "masked.npy", np.concatenate(masked))
    np.save(out / "full.npy", np.concatenate(full))
    np.save(out / "val.npy", keep >= int(total * 0.9))
    print(f"done: {len(keep)} samples ({(keep >= int(total * 0.9)).sum()} validation) in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--max", type=int, default=8000)
    a = ap.parse_args()
    main(a.source, a.max)
