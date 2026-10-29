# fast-lipsync

Lip-sync a talking-head video to a new voice track, at close to real time on an Apple Silicon Mac. It is built on
MuseTalk 1.5, with the footage-side work cached and the hot path moved to Core ML.

```sh
./setup.sh                                                 # once: venv, MuseTalk, weights, Core ML UNet
./lipsync.sh footage.mp4 voice.wav out.mp4 --start 12.5     # one continuous take from 12.5 s
./lipsync.sh footage.mp4 voice.wav out.mp4 --timeline cuts.json
```

`cuts.json` stitches several takes: `[{"src_start": 7.0, "t0": 0.0, "t1": 4.03}, ...]` means that output
seconds 0–4.03 show the footage from 7.0 s on. The cuts must cover the whole voice track. Output is 25 fps H.264
(hardware encoder, 24 Mb/s) with the voice muxed in.

## Speed (M4 Pro, 1080×1920 footage, warm cache)

| stage | cost |
|---|---|
| model loading | ~4.3 s per run |
| generated mouths (UNet + decode + blend + encode) | ~21.5 output frames/s |
| real mouths | ~25 output frames/s (UNet runs on every 2nd frame) |

The first run on new footage also prepares each source frame it uses: about 0.13 s per frame. That happens once, because the cache is keyed by the footage file and only frames that are
missing get computed.

## Why it is fast

Stock MuseTalk detects faces, crops, VAE-encodes, runs the UNet, VAE-decodes and face-parses every frame on
every run. Here, each step is either cached or replaced with something cheaper:

1. **Footage-side work is cached per source frame** (`tools/prepare.py` → `cache/avatar/<stem>/`): face
   boxes, VAE latents of the masked and reference crops, blend masks and mouth openness. None of it depends on
   the audio, so a new voice track reuses all of it.
2. **InsightFace instead of mmpose/DWPose** for face boxes, because mmpose does not build on macOS. It runs on
   the CoreML execution provider at 320 px detection size, and boxes are smoothed over 5 frames within each cut.
3. **UNet converted to Core ML** (`tools/convert_unet_coreml.py`): fp16, fixed batch of 8, `CPU_AND_GPU`.
   This is the bottleneck, at about 30 fps alone.
4. **TAESD decoder instead of the SD VAE decoder**, about 17× faster on MPS. TAESD decodes the UNet's
   SD-scaled latents directly.
5. **Mouth-only blend mask**, a feathered oval cached at 256 px, instead of MuseTalk's face-parsing jaw mask.
   This is cheaper and keeps the real moustache and beard. The generated mouth is colour-matched to the frame,
   gets its fine texture back and is sharpened (`refine_mouth`).
6. **Pipelined output.** Blending runs on a 6-thread pool (OpenCV releases the GIL) and frames stream straight
   into ffmpeg's `h264_videotoolbox` while the UNet works on the next batch.
7. **Open-mouth reference.** Each cut's most open-mouth frame is the UNet's reference image, so teeth show
   (`--ref same` turns this off).

## Real-mouth mode (`--mouth real`)

MuseTalk's mouths can be soft, toothless or the wrong colour. In this mode MuseTalk only supplies the mouth
*shape*, and every mouth pixel comes from the speaker's real footage:

```sh
.venv/bin/python -I tools/mouth_library.py long-raw-footage.mp4   # once; takes ~0.7x the footage length
./lipsync.sh footage.mp4 voice.wav out.mp4 --mouth real --library long-raw-footage
```

- `mouth_library.py` takes every 2nd frame and records InsightFace 106 landmarks, a mouth patch warped to a
  canonical frame (eyes and nose bottom fixed), and a lip-shape descriptor. Frames with a hand near the mouth
  are flagged by MediaPipe and never used.
- `real_mouth.py` takes MuseTalk's shape and runs it at every 2nd frame, interpolating within a shot. It
  queries the K nearest real mouths, then uses Viterbi to pick a path: continuing a real run of footage is
  free and jumping costs `SWITCH_COST`. The chosen mouth is pasted with a 1-frame cross-fade, colour-matched
  and feathered.

`--library` defaults to the footage's file stem (`cache/mouths/<stem>/`).

## Person-specific fine-tune (experimental)

`tools/train_data.py footage.mp4` builds a training set from the mouth library's usable frames.
`tools/finetune.py footage.mp4` then trains a LoRA on the UNet (latent L1 + TAESD-decoded pixel L1 + VGG
perceptual loss), at ~1.3 s/step on an M4 Pro (~1 h for the default 3000 steps).
To use a result, convert it and the Core ML loader picks it up for that footage automatically:

```sh
.venv/bin/python -I tools/convert_unet_coreml.py 8 --weights models/musetalk_ft/<stem>/unet.pth --name <stem>
```

## Layout

| path | what |
|---|---|
| `tools/lipsync.py` | the runtime: audio features → Core ML UNet → TAESD → blend → ffmpeg |
| `tools/prepare.py` | footage cache (called automatically; run directly to warm it) |
| `tools/convert_unet_coreml.py` | MuseTalk UNet → Core ML package |
| `tools/mouth_library.py`, `tools/real_mouth.py` | real-mouth mode |
| `tools/train_data.py`, `tools/finetune.py` | LoRA fine-tune |
| `patches/musetalk-mps.patch` | lets MuseTalk's face parser run off-CUDA |

Known limits: MuseTalk's 256 px crop erases facial hair inside the mouth region in generated mode, and the
footage must hold one dominant front-facing face.
