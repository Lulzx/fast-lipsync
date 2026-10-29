"""Faster-than-real-time lip-sync with MuseTalk 1.5 on Apple Silicon.

Per run this only does: Whisper-tiny audio features -> MuseTalk UNet (Core ML) -> TAESD decode (MPS) -> blend.
Everything that depends only on the footage (face boxes, crop latents, blend masks) comes from the cache built
by tools/prepare.py, which runs automatically for any frames not cached yet. Face boxes come from InsightFace
instead of MuseTalk's mmpose/DWPose, which does not build on macOS.

Usage:
  python tools/lipsync.py footage.mp4 voice.wav out.mp4 [--start 12.5]
  python tools/lipsync.py footage.mp4 voice.wav out.mp4 --timeline cuts.json
      cuts.json = [{"src_start": 7.0, "t0": 0.0, "t1": 4.03}, ...]: output seconds t0..t1 show the footage from
      src_start on. Cuts must cover the whole voice track.
  [--mouth generated|real] [--decoder taesd|vae] [--unet coreml|mps] [--ref open|same]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MT = ROOT / "vendor/MuseTalk"
sys.path.insert(0, str(MT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tools (prepare, real_mouth, mouth_library)

from musetalk.models.vae import VAE  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_crop_box  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16
FPS = 25
BATCH = 8           # matches the fixed batch of the Core ML UNet
EXTRA_MARGIN = 10   # MuseTalk 1.5 extends the crop below the chin by this many px
SMOOTH = 5          # frames in the moving average that steadies the face box
COLOR_MATCH = 0.8    # how far the generated mouth's colour is pulled to the real frame (0-1)
DETAIL = 0.9         # amount of the real frame's fine texture added back onto the generated mouth
SHARPEN = 0.6        # unsharp-mask strength on the upscaled 256px mouth
SHAPE_STEP = 2       # real-mouth mode: run MuseTalk on every 2nd frame, interpolate the shape in between
COREML_UNET = ROOT / "models/coreml/musetalk_unet_b8.mlpackage"
CACHE = ROOT / "cache"
# Apple's hardware H.264 encoder at a high bitrate; re-encode with libx264 if the result is the final delivery
HW_ENC = ["-c:v", "h264_videotoolbox", "-b:v", "24M", "-pix_fmt", "yuv420p"]


# ---------- footage-side helpers (also used by prepare.py) ----------

def load_vae():
    vae = VAE(model_path=str(MT / "models/sd-vae"), use_float16=True)
    vae.vae.to(DEVICE)
    vae.device = DEVICE
    return vae


def load_face_detector():
    from insightface.app import FaceAnalysis
    face = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], root=str(ROOT / "models/insightface"),
                        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    face.prepare(ctx_id=-1, det_size=(320, 320))
    return face


def detect_face(frame, face):
    """MuseTalk-style crop box, the two mouth corners and mouth openness:
    (x1, y1, x2, y2, mlx, mly, mrx, mry, open)."""
    s = 640 / max(frame.shape[:2])
    faces = face.get(cv2.resize(frame, None, fx=s, fy=s))
    if not faces:
        return None
    fc = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    lm, kps = fc.landmark_2d_106 / s, fc.kps / s
    x1, x2 = lm[:33, 0].min(), lm[:33, 0].max()
    chin = lm[:33, 1].max()
    eye_y, nose_y = kps[:2, 1].mean(), kps[2, 1]
    half = eye_y + 0.67 * (nose_y - eye_y)  # ~ landmark 29 of the 68-point layout
    # InsightFace 2d106: 62 = upper inner lip, 60 = lower inner lip; openness relative to mouth width
    opening = (lm[60, 1] - lm[62, 1]) / max(kps[4, 0] - kps[3, 0], 1)
    return (x1, max(0, half - (chin - half)), x2, chin, *kps[3], *kps[4], opening)


def smooth_faces(raw, height, width=1080):
    """Fill missed detections and average over SMOOTH frames so the crop and mask don't jitter."""
    first = next(b for b in raw if b is not None)
    filled, last = [], first
    for b in raw:
        last = b if b is not None else last
        filled.append(last)
    arr = np.array(filled, dtype=np.float32)
    pad = np.pad(arr, ((SMOOTH // 2, SMOOTH // 2), (0, 0)), mode="edge")
    k = np.ones(SMOOTH) / SMOOTH
    arr = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(arr.shape[1])], 1)
    boxes = [(max(0, int(r[0])), max(0, int(r[1])), min(width, int(r[2])), min(height, int(r[3]) + EXTRA_MARGIN))
             for r in arr]
    return boxes, arr[:, 4:8], arr[:, 8]


def mouth_mask(box, corners):
    """Feathered oval around the mouth, in crop-box coordinates; the rest of the face (beard) stays original."""
    crop_box, _ = get_crop_box(box, 1.5)
    xs, ys, xe, ye = crop_box
    mask = np.zeros((ye - ys, xe - xs), np.uint8)
    (lx, ly), (rx, ry) = corners.reshape(2, 2)
    w = rx - lx
    # top edge sits just above the upper lip so the real moustache is kept; the oval extends down over the chin
    cx, cy = (lx + rx) / 2 - xs, (ly + ry) / 2 - ys + 0.30 * w
    cv2.ellipse(mask, (int(cx), int(cy)), (int(w * 0.85), int(w * 0.52)), 0, 0, 360, 255, -1)
    k = int(w * 0.3) // 2 * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0), crop_box


@torch.no_grad()
def encode_crops(frames, boxes, vae):
    """VAE latents for UNet input: masked lower-half crop + full reference crop."""
    masked, ref = [], []
    for f, (x1, y1, x2, y2) in zip(frames, boxes):
        crop = cv2.resize(f[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        masked.append(vae.preprocess_img(crop, half_mask=True))
        ref.append(vae.preprocess_img(crop, half_mask=False))
    m = vae.encode_latents(torch.cat(masked).to(DEVICE))
    r = vae.encode_latents(torch.cat(ref).to(DEVICE))
    return torch.cat([m, r], 1)


def source_indices(timeline, src_fps):
    """Source frame index for every 25 fps output frame, cut by cut."""
    idx = []
    for s in timeline:
        n = round(s["t1"] * FPS) - round(s["t0"] * FPS)
        idx += [int(round((s["src_start"] + k / FPS) * src_fps)) for k in range(n)]
    return idx


def read_source_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    frames, pos, cur = {}, -1, None
    for want in sorted(set(indices)):
        if want < pos or want - pos > 90:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            pos = want - 1
        while pos < want:
            ok, cur = cap.read()
            pos += 1
        frames[want] = cur
    return [frames[i] for i in indices]


# ---------- per run ----------

class Timer:
    def __init__(self):
        self.t, self.marks = time.time(), []

    def mark(self, name):
        now = time.time()
        self.marks.append((name, now - self.t))
        self.t = now

    def report(self, clip_seconds):
        total = sum(d for _, d in self.marks)
        for name, d in self.marks:
            print(f"  {name:16s} {d:6.2f}s")
        print(f"  {'total':16s} {total:6.2f}s for {clip_seconds:.1f}s of video ({clip_seconds / total:.2f}x real time)")


class CoreMLUNet:
    def __init__(self, stem=None):
        """Uses the person-specific fine-tune for this footage (tools/finetune.py) when one has been converted."""
        import coremltools as ct
        ft = ROOT / f"models/coreml/musetalk_unet_{stem}_b{BATCH}.mlpackage"
        path = ft if stem and ft.exists() else COREML_UNET
        print(f"  unet: {path.name}")
        self.m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    def __call__(self, latents, audio):
        n = latents.shape[0]
        if n < BATCH:  # the model has a fixed batch; pad the last one
            latents = np.concatenate([latents, np.repeat(latents[-1:], BATCH - n, 0)])
            audio = np.concatenate([audio, np.repeat(audio[-1:], BATCH - n, 0)])
        out = self.m.predict({"latents": latents.astype(np.float32), "audio": audio.astype(np.float32)})["pred"]
        return torch.from_numpy(out[:n])


class MPSUNet:
    def __init__(self):
        from musetalk.models.unet import UNet
        u = UNet(unet_config=str(MT / "models/musetalkV15/musetalk.json"),
                 model_path=str(MT / "models/musetalkV15/unet.pth"), device=torch.device("cpu"))
        self.m = u.model.to(DEVICE, DTYPE).eval()
        self.t = torch.tensor([0], device=DEVICE)

    @torch.no_grad()
    def __call__(self, latents, audio):
        return self.m(torch.from_numpy(latents).to(DEVICE, DTYPE), self.t,
                      encoder_hidden_states=torch.from_numpy(audio).to(DEVICE, DTYPE)).sample


class Decoder:
    """TAESD (tiny, ~17x faster on MPS) or the full SD VAE decoder MuseTalk was trained with."""

    def __init__(self, kind):
        self.kind = kind
        if kind == "taesd":
            from diffusers import AutoencoderTiny
            self.m = AutoencoderTiny.from_pretrained(str(ROOT / "models/taesd")).to(DEVICE, DTYPE).eval()
        else:
            self.vae = load_vae()

    @torch.no_grad()
    def __call__(self, pred):
        if self.kind == "taesd":
            # TAESD decodes SD-scaled latents directly (its scaling_factor is 1.0), same as the UNet emits
            img = self.m.decode(pred.to(DEVICE, DTYPE)).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            img = (img.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
            return img[..., ::-1]
        return self.vae.decode_latents(pred.to(DEVICE, DTYPE))


def refine_mouth(gen, src):
    """Make the 256px generated face hold up at 1080p: match the real frame's colour, borrow its fine texture
    (beard, pores) and sharpen. gen and src are same-size BGR crops of the face box."""
    lab_g = cv2.cvtColor(gen, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    lower = slice(lab_g.shape[0] * 45 // 100, None)  # stats from the mouth half only
    mg, sg = lab_g[lower].mean((0, 1)), lab_g[lower].std((0, 1)) + 1e-3
    ms, ss = lab_s[lower].mean((0, 1)), lab_s[lower].std((0, 1)) + 1e-3
    lab_g = (lab_g - mg) / sg * (COLOR_MATCH * ss + (1 - COLOR_MATCH) * sg) + COLOR_MATCH * ms + (1 - COLOR_MATCH) * mg
    g = cv2.cvtColor(np.clip(lab_g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    s = src.astype(np.float32)
    g += DETAIL * (s - cv2.GaussianBlur(s, (0, 0), 0.9))      # real fine texture only (hair, pores), not lip edges
    g += SHARPEN * (g - cv2.GaussianBlur(g, (0, 0), 1.2))     # unsharp mask
    return np.clip(g, 0, 255).astype(np.uint8)


def blend(frame, face_img, box, mask_small, crop_box):
    x1, y1, x2, y2 = box
    xs, ys, xe, ye = crop_box
    H, W = frame.shape[:2]
    mask = cv2.resize(mask_small, (xe - xs, ye - ys), interpolation=cv2.INTER_LINEAR)
    face_img = cv2.resize(face_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    face_img = refine_mouth(face_img, frame[y1:y2, x1:x2])
    cx0, cy0, cx1, cy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)  # crop box can run off-frame
    region = frame[cy0:cy1, cx0:cx1]
    large = region.copy()
    large[y1 - cy0:y2 - cy0, x1 - cx0:x2 - cx0] = face_img
    m = mask[cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs, None].astype(np.float32) / 255
    out = frame.copy()
    out[cy0:cy1, cx0:cx1] = (large * m + region * (1 - m)).astype(np.uint8)
    return out


def reference_frames(timeline, idx, row, openness):
    """For each output frame, the source frame in the same cut with the most open mouth. MuseTalk copies
    teeth and lip detail from its reference image, so a closed-mouth reference gives closed, toothless mouths."""
    refs, k = [], 0
    for s in timeline:
        n = round(s["t1"] * FPS) - round(s["t0"] * FPS)
        cut = idx[k:k + n]
        best = max(cut, key=lambda src: openness[row[src]])
        refs += [best] * len(cut)
        k += n
    return refs


def real_mouth_frames(frames, gen, n, idx, row, boxes, masks, crop_boxes, library, pool):
    """Replace each generated mouth with the closest real mouth from the speaker's footage (see real_mouth.py).

    gen maps frame number -> generated face for a subset of frames (every SHAPE_STEP-th); mouth shape and
    placement for the frames in between are interpolated within the same shot."""
    import real_mouth as RM
    import mouth_library as ML
    lib = RM.Library(library)
    face = load_face_detector()
    H, W = frames[0].shape[:2]

    def region(i):
        xs, ys, xe, ye = crop_boxes[row[idx[i]]]
        return max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)

    def target(i):
        """Landmarks of the lip-synced frame: the mouth shape we want, in frame coordinates."""
        s = row[idx[i]]
        x0, y0, x1, y1 = region(i)
        img = blend(frames[i], gen[i], boxes[s], masks[s], crop_boxes[s])[y0:y1, x0:x1]
        faces = face.get(img)
        if not faces:
            return None
        fc = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return fc.landmark_2d_106 + (x0, y0), fc.kps + (x0, y0)

    keys = sorted(gen)
    found = dict(zip(keys, pool.map(target, keys)))
    sampled = [k for k in keys if found[k] is not None]

    def same_shot(a, b):
        return abs(idx[a] - idx[b]) <= 2 * abs(a - b) + 1

    targets = []
    for i in range(n):
        j = np.searchsorted(sampled, i)
        a, b = sampled[max(j - 1, 0)], sampled[min(j, len(sampled) - 1)]
        if a == b or not same_shot(a, b):   # at a cut (or an edge), use whichever neighbour is in this shot
            k = a if same_shot(a, i) else b
            targets.append(found[k])
            continue
        w = (i - a) / (b - a)
        targets.append(tuple(found[a][q] * (1 - w) + found[b][q] * w for q in range(2)))
    targets = RM.smooth_targets(targets, idx[:n])
    descs = [ML.descriptor(lm, ML.to_canon(lm, kps)) for lm, kps in targets]
    choice = RM.choose(lib, descs)
    # frames since the last jump to another moment of the footage, for cross-fading
    jumps = [0] + [k for k in range(1, len(choice))
                   if choice[k] not in (choice[k - 1], choice[k - 1] + 1)]
    since = np.zeros(len(choice), int) + RM.FADE
    for k in jumps[1:]:
        since[k:k + RM.FADE] = np.arange(min(RM.FADE, len(choice) - k))

    def render(i):
        lm, kps = targets[i]
        patch = lib.patch(choice[i])
        if since[i] < RM.FADE:  # cross-fade from the mouth that was showing before the jump
            w = (since[i] + 1) / (RM.FADE + 1)
            patch = cv2.addWeighted(patch, w, lib.patch(choice[i - since[i] - 1]), 1 - w, 0)
        return RM.paste(frames[i], lm, kps, patch, lib.mask, region(i))

    print(f"  real mouths: {len(jumps)} runs over {len(choice)} frames")
    return pool.map(render, range(n))


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def make_timeline(source, voice, start=0.0, timeline=None):
    """The cuts to lip-sync: a JSON timeline, or one continuous take from `start` as long as the voice."""
    if timeline:
        return json.loads(Path(timeline).read_text())
    length = duration(voice)
    if start + length > duration(source):
        sys.exit(f"{source} has no {length:.1f}s of footage after {start}s; pass an earlier --start or a --timeline")
    return [{"src_start": start, "t0": 0.0, "t1": round(length, 2)}]


def main(source, voice, out, timeline, decoder="taesd", unet_kind="coreml", ref_mode="open", mouth="generated",
         library=None):
    import prepare
    source = Path(source)
    t = Timer()
    src_fps = cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FPS)
    idx = source_indices(timeline, src_fps)
    prepare.ensure(source, idx)  # no-op when every frame is cached
    t.mark("prepare (cache)")

    avatar = CACHE / "avatar" / source.stem
    row = {s: r for r, s in enumerate(np.load(avatar / "indices.npy").tolist())}  # source frame -> cache row
    boxes = np.load(avatar / "boxes.npy")
    latents = np.load(avatar / "latents.npy")
    masks = np.load(avatar / "masks.npy")
    crop_boxes = np.load(avatar / "crop_boxes.npy")
    openness = np.load(avatar / "openness.npy")
    unet = CoreMLUNet(source.stem) if unet_kind == "coreml" else MPSUNet()
    dec = Decoder(decoder)
    from transformers import WhisperModel
    whisper = WhisperModel.from_pretrained(str(MT / "models/whisper")).to(DEVICE, DTYPE).eval()
    audio = AudioProcessor(feature_extractor_path=str(MT / "models/whisper"))
    t.mark("load models")

    frames = read_source_frames(source, idx)
    t.mark("read frames")

    with torch.no_grad():
        feats, length = audio.get_audio_feature(str(voice), weight_dtype=DTYPE)
        chunks = audio.get_whisper_chunk(feats, DEVICE, DTYPE, whisper, length, fps=FPS,
                                         audio_padding_length_left=2, audio_padding_length_right=2)
    chunks = torch.stack(list(chunks)).float().cpu().numpy()
    pe = __import__("musetalk.models.unet", fromlist=["PositionalEncoding"]).PositionalEncoding(d_model=384)
    with torch.no_grad():
        chunks = pe(torch.from_numpy(chunks)).numpy()
    n = min(len(idx), len(chunks))
    t.mark("audio features")

    # blend + encode in a worker thread while the UNet runs
    h, w = frames[0].shape[:2]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-i", str(voice), "-map", "0:v", "-map", "1:a",
                            *HW_ENC, "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    q = queue.Queue(maxsize=4)
    pool = ThreadPoolExecutor(6)  # OpenCV releases the GIL, so blending scales across cores

    def blend_one(i, face_img):
        s = row[idx[i]]
        return blend(frames[i], face_img, boxes[s], masks[s], crop_boxes[s])

    def writer():
        while (item := q.get()) is not None:
            i0, faces = item
            for img in pool.map(blend_one, range(i0, i0 + len(faces)), faces):
                enc.stdin.write(img.tobytes())

    refs = reference_frames(timeline, idx, row, openness) if ref_mode == "open" else idx

    def generate(ks):
        # UNet input = masked latent of this frame + reference latent (this frame, or the cut's most open mouth)
        lat = np.stack([np.concatenate([latents[row[idx[k]]][:4], latents[row[refs[k]]][4:]]) for k in ks])
        return dec(unet(lat, chunks[ks]))

    if mouth == "real":
        # only the mouth shape is used, and it changes slowly enough to sample every SHAPE_STEP-th frame
        ks = sorted(set(range(0, n, SHAPE_STEP)) | {n - 1})
        gen = {k: f for c in range(0, len(ks), BATCH) for k, f in zip(ks[c:c + BATCH], generate(ks[c:c + BATCH]))}
        t.mark("unet+decode")
        for img in real_mouth_frames(frames, gen, n, idx, row, boxes, masks, crop_boxes, library or source.stem, pool):
            enc.stdin.write(img.tobytes())
    else:
        th = threading.Thread(target=writer)
        th.start()
        for i in range(0, n, BATCH):
            q.put((i, generate(list(range(i, min(i + BATCH, n))))))
        q.put(None)
        th.join()
    enc.stdin.close()
    enc.wait()
    pool.shutdown()
    t.mark("mouths+encode" if mouth == "real" else "unet+decode+blend")
    t.report(n / FPS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="footage of the speaker (any fps; output is 25 fps)")
    ap.add_argument("audio", help="voice track to lip-sync to")
    ap.add_argument("out", help="output .mp4 (video + the voice track)")
    ap.add_argument("--start", type=float, default=0.0, help="footage second to start from (single take)")
    ap.add_argument("--timeline", help="JSON list of cuts {src_start, t0, t1}; overrides --start")
    ap.add_argument("--decoder", default="taesd", choices=["taesd", "vae"])
    ap.add_argument("--unet", default="coreml", choices=["coreml", "mps"])
    ap.add_argument("--ref", default="open", choices=["open", "same"],
                    help="reference image: the cut's most open-mouth frame (shows teeth) or the frame itself")
    ap.add_argument("--mouth", default="generated", choices=["real", "generated"],
                    help="real: paste the closest real mouth from the footage (needs tools/mouth_library.py); "
                         "generated: MuseTalk's own mouth")
    ap.add_argument("--library", help="mouth library name for --mouth real (default: the video's file stem)")
    a = ap.parse_args()
    tl = make_timeline(a.video, a.audio, a.start, a.timeline)
    main(a.video, a.audio, a.out, tl, a.decoder, a.unet, a.ref, a.mouth, a.library)
