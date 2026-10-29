"""Prepare the footage side of lip-sync for the source frames a run uses.

For each needed source frame: MuseTalk face box, VAE latents of the face crop (masked + reference), a feathered
mouth-only blend mask (MuseTalk's own jaw mask smears beards) and the mouth openness. None of this depends on the
voice, so it is cached per footage file and only missing frames are computed. lipsync.py calls ensure() itself;
run this directly to warm the cache ahead of time.

Usage: python tools/prepare.py footage.mp4 [--start 0] [--duration 60]
Cache: cache/avatar/<footage stem>/{indices, boxes, latents, masks, crop_boxes, openness}.npy
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lipsync as L  # noqa: E402

MASK_RES = 256  # blend masks are heavily blurred, so they are stored small and upscaled at blend time
FIELDS = ("indices", "boxes", "latents", "masks", "crop_boxes", "openness")


def load_cache(d):
    if not all((d / f"{k}.npy").exists() for k in FIELDS):
        return None
    return {k: np.load(d / f"{k}.npy") for k in FIELDS}


def runs(indices):
    """Split sorted frame indices into contiguous runs so box smoothing never crosses a cut."""
    out = [[indices[0]]]
    for i in indices[1:]:
        if i - out[-1][-1] <= 2:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def ensure(source, indices):
    """Make sure every source frame in indices is in the cache."""
    t0 = time.time()
    source = Path(source)
    out = L.CACHE / "avatar" / source.stem
    out.mkdir(parents=True, exist_ok=True)
    cache = load_cache(out)
    have = set(cache["indices"].tolist()) if cache else set()
    need = sorted(set(indices) - have)
    if not need:
        return
    print(f"preparing {len(need)} new frames ({len(have)} cached)")
    with torch.no_grad():
        prepare(need, source, out, cache, t0)


def prepare(need, source, out, cache, t0):
    cap = cv2.VideoCapture(str(source))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vae, face = L.load_vae(), L.load_face_detector()
    frames = dict(zip(need, L.read_source_frames(source, need)))

    boxes, corners, opening = {}, {}, {}
    for run in runs(need):
        bs, cs, os_ = L.smooth_faces([L.detect_face(frames[i], face) for i in run], height, width)
        for i, b, c, o in zip(run, bs, cs, os_):
            boxes[i], corners[i], opening[i] = b, c, o

    latents, masks, crop_boxes = [], [], []
    for k in range(0, len(need), L.BATCH):
        chunk = need[k:k + L.BATCH]
        latents.append(L.encode_crops([frames[i] for i in chunk], [boxes[i] for i in chunk], vae)
                       .cpu().numpy().astype(np.float16))
        for i in chunk:
            mask, crop_box = L.mouth_mask(boxes[i], corners[i])
            masks.append(cv2.resize(mask, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA))
            crop_boxes.append(crop_box)

    new = {"indices": np.array(need, dtype=np.int32), "boxes": np.array([boxes[i] for i in need], dtype=np.int32),
           "latents": np.concatenate(latents), "masks": np.stack(masks),
           "crop_boxes": np.array(crop_boxes, dtype=np.int32),
           "openness": np.array([opening[i] for i in need], dtype=np.float32)}
    if cache:
        new = {k: np.concatenate([cache[k], new[k]]) for k in FIELDS}
    for k in FIELDS:
        np.save(out / f"{k}.npy", new[k])
    print(f"prepared in {time.time() - t0:.0f}s -> {out} ({len(new['indices'])} frames cached)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, help="seconds to prepare (default: to the end of the footage)")
    a = ap.parse_args()
    dur = a.duration or L.duration(a.video) - a.start - 0.1  # stay clear of the last frame
    src_fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
    ensure(a.video, L.source_indices([{"src_start": a.start, "t0": 0.0, "t1": dur}], src_fps))
