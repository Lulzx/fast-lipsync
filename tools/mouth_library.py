"""Build a library of the speaker's real mouths from raw footage, for the real-mouth lip-sync mode.

For every STEP-th frame: InsightFace 106 landmarks, a similarity transform that puts the eyes and nose bottom at
fixed canonical points (they don't move when talking), the mouth patch warped into that canonical frame (JPEG),
a mouth-shape descriptor (lip landmarks in canonical coordinates, plus a little head-pose information), and a flag
for frames where a hand (and so usually the mic) is near the mouth; those are never used.

Usage: python tools/mouth_library.py footage.mp4
Writes cache/mouths/<source stem>/{frames, desc, ok, patches}.npy
"""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lipsync as L  # noqa: E402

STEP = 2                       # every 2nd frame (12.5 per second of footage)
PATCH_W, PATCH_H = 256, 208    # canonical mouth patch
# canonical anchor positions: left eye, right eye, nose bottom (InsightFace 2d106 point 80)
CANON = np.float32([[78, -44], [178, -44], [128, 26]])
LIPS = list(range(52, 72))     # InsightFace 2d106 mouth points
POSE = [86, 0]                 # nose tip, chin: coarse head-pose cue
POSE_WEIGHT = 0.5


def anchors(lm, kps):
    return np.float32([kps[0], kps[1], lm[80]])


def to_canon(lm, kps):
    """2x3 similarity transform from frame coordinates into the canonical mouth patch."""
    m, _ = cv2.estimateAffinePartial2D(anchors(lm, kps), CANON)
    return m


def descriptor(lm, m):
    pts = cv2.transform(lm[None].astype(np.float32), m)[0]
    lips = pts[LIPS].ravel()
    pose = pts[POSE].ravel() * POSE_WEIGHT
    return np.concatenate([lips, pose]).astype(np.float32)


def landmarks(frame, face):
    s = 640 / max(frame.shape[:2])
    faces = face.get(cv2.resize(frame, None, fx=s, fy=s))
    if not faces:
        return None, None
    fc = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return fc.landmark_2d_106 / s, fc.kps / s


def main(src):
    src = Path(src)
    out = L.ROOT / "cache/mouths" / src.stem
    out.mkdir(parents=True, exist_ok=True)
    face = L.load_face_detector()
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(L.ROOT / "models/hand_landmarker.task")), num_hands=2))
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, descs, oks, patches = [], [], [], []
    t0 = time.time()
    i = -1
    while True:
        if not cap.grab():
            break
        i += 1
        if i % STEP:
            continue
        _, f = cap.retrieve()
        lm, kps = landmarks(f, face)
        if lm is None:
            continue
        m = to_canon(lm, kps)
        # hand near the mouth -> occluded (mic or fingers would get pasted)
        h, w = f.shape[:2]
        small = cv2.cvtColor(cv2.resize(f, (w // 3, h // 3)), cv2.COLOR_BGR2RGB)
        res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=small))
        mouth = lm[LIPS]
        mw = np.ptp(mouth[:, 0])
        lo, hi = mouth.min(0) - mw * 0.6, mouth.max(0) + mw * 0.6
        occluded = any(((np.array([[p.x * w, p.y * h] for p in hl]) > lo) & (np.array([[p.x * w, p.y * h] for p in hl]) < hi)).all(1).any()
                       for hl in res.hand_landmarks)
        patch = cv2.warpAffine(f, m, (PATCH_W, PATCH_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(i)
        descs.append(descriptor(lm, m))
        oks.append(not occluded)
        patches.append(np.frombuffer(cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 93])[1], np.uint8))
        if len(frames) % 1000 == 0:
            print(f"  {i}/{n} frames  ({sum(oks)} usable, {time.time() - t0:.0f}s)")
    np.save(out / "frames.npy", np.array(frames, np.int32))
    np.save(out / "desc.npy", np.stack(descs))
    np.save(out / "ok.npy", np.array(oks))
    np.save(out / "patches.npy", np.array(patches, dtype=object), allow_pickle=True)
    print(f"{len(frames)} mouths, {sum(oks)} usable (no hand near the mouth), {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
