"""Real-mouth rendering: replace each lip-synced mouth with the closest real mouth from the speaker's footage.

MuseTalk decides the mouth shape for each frame of audio; this module finds the real mouth (from a library built
by mouth_library.py) whose shape matches best, keeps the sequence smooth with a Viterbi pass, and pastes it onto
the original frame. Every mouth pixel in the result is real footage: teeth, lips, beard.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

import mouth_library as ML

K = 12              # nearest real mouths considered per frame
OPEN_GAIN = 1.25    # MuseTalk under-opens; ask the library for slightly more open mouths
SWITCH_COST = 4.0   # cost of jumping to another moment of the footage, in units of the median match distance
FADE = 1            # frames to cross-fade at a jump (longer fades ghost two mouths into a double lip)
BEAM = 40           # candidates kept per frame (keeps Viterbi linear in clip length)
MASK_CENTER, MASK_AXES, MASK_FEATHER = (128, 84), (102, 76), 9   # canonical patch px; must fully cover the original mouth


class Library:
    def __init__(self, stem):
        d = ML.L.ROOT / "cache/mouths" / stem
        ok = np.load(d / "ok.npy")
        self.frames = np.load(d / "frames.npy")[ok]
        self.desc = np.load(d / "desc.npy")[ok]
        self.patches = np.load(d / "patches.npy", allow_pickle=True)[ok]
        self.tree = cKDTree(self.desc)
        mask = np.zeros((ML.PATCH_H, ML.PATCH_W), np.float32)
        cv2.ellipse(mask, MASK_CENTER, MASK_AXES, 0, 0, 360, 1.0, -1)
        self.mask = cv2.GaussianBlur(mask, (0, 0), MASK_FEATHER)

    def patch(self, j):
        return cv2.imdecode(self.patches[j], cv2.IMREAD_COLOR)


def open_more(desc):
    """Stretch lip points vertically about the mouth-corner line."""
    d = desc.copy()
    lips = d[:len(ML.LIPS) * 2].reshape(-1, 2)
    cy = lips[[0, 9], 1].mean()  # 52 and 61 are the mouth corners (index 0 and 9 within 52..71)
    lips[:, 1] = cy + (lips[:, 1] - cy) * OPEN_GAIN
    return d


def choose(lib, target_descs):
    """Pick a library mouth per frame, preferring to keep playing a real run of footage.

    Candidates per frame: the K nearest mouths, plus the continuations (same frame held, or the next sampled
    frame) of the previous frame's candidates. Continuing costs nothing; jumping elsewhere costs SWITCH_COST.
    Solved exactly with Viterbi."""
    q = np.stack([open_more(d) for d in target_descs])
    _, near = lib.tree.query(q, k=K)
    scale = np.median(np.linalg.norm(lib.desc[near[:, 0]] - q, axis=1)) + 1e-6
    switch = SWITCH_COST * scale
    nxt = np.append(lib.frames[1:] == lib.frames[:-1] + ML.STEP, False)  # library row j+1 continues row j

    cands = [near[0]]
    cost = np.linalg.norm(lib.desc[near[0]] - q[0], axis=1)
    back = []
    for t in range(1, len(q)):
        prev = cands[-1]
        cont = np.concatenate([prev, prev[nxt[prev]] + 1])
        cur = np.unique(np.concatenate([near[t], cont]))
        match = np.linalg.norm(lib.desc[cur] - q[t], axis=1)
        # transition [cur, prev]: free if cur holds prev or is its next frame, else a switch
        stay = (cur[:, None] == prev[None]) | ((cur[:, None] == prev[None] + 1) & nxt[prev][None])
        total = cost[None] + np.where(stay, 0.0, switch)
        arg, cost = total.argmin(1), total.min(1) + match
        keep = np.argsort(cost)[:BEAM]
        back.append(arg[keep])
        cost = cost[keep]
        cands.append(cur[keep])
    k = int(cost.argmin())
    path = [int(cands[-1][k])]
    for t in range(len(q) - 1, 0, -1):
        k = int(back[t - 1][k])
        path.append(int(cands[t - 1][k]))
    return path[::-1]


def smooth_targets(targets, idx, window=5):
    """Average landmarks over neighbouring frames of the same shot so the pasted mouth doesn't wobble."""
    lm = np.stack([t[0] for t in targets])
    kps = np.stack([t[1] for t in targets])
    cut = np.r_[0, np.flatnonzero(np.abs(np.diff(idx)) > 2) + 1, len(idx)]
    out_lm, out_kps = lm.copy(), kps.copy()
    for a, b in zip(cut[:-1], cut[1:]):
        for i in range(a, b):
            lo, hi = max(a, i - window // 2), min(b, i + window // 2 + 1)
            out_lm[i], out_kps[i] = lm[lo:hi].mean(0), kps[lo:hi].mean(0)
    return list(zip(out_lm, out_kps))


def paste(frame, lm, kps, patch, mask, region):
    """Warp a canonical real-mouth patch onto the frame (inside region), colour-matched and feathered."""
    x0, y0, x1, y1 = region
    inv = cv2.invertAffineTransform(ML.to_canon(lm, kps))
    inv[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(patch, inv, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, inv, size, flags=cv2.INTER_LINEAR)[..., None]
    target = frame[y0:y1, x0:x1]
    # match colour to this frame (lighting differs slightly across the footage)
    sel = m[..., 0] > 0.5
    if sel.sum() > 50:
        wl = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        tl = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        mw, sw = wl[sel].mean(0), wl[sel].std(0) + 1e-3
        mt, st = tl[sel].mean(0), tl[sel].std(0) + 1e-3
        wl = (wl - mw) / sw * (0.5 * st + 0.5 * sw) + mt
        warped = cv2.cvtColor(np.clip(wl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = frame.copy()
    out[y0:y1, x0:x1] = (warped * m + target * (1 - m)).astype(np.uint8)
    return out
