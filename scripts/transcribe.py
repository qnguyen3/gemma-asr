#!/usr/bin/env python
"""Transcribe audio files of any length with a trained checkpoint.

    python scripts/transcribe.py --ckpt runs/vi/best.pt audio1.wav audio2.mp3

Long audio is cut into <= --chunk_seconds pieces at the quietest point near each boundary,
and all chunks are decoded in batches.
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.data import collate  # noqa: E402
from gemma_asr.model import GemmaCTC, greedy_decode  # noqa: E402
from gemma_asr.units import Units  # noqa: E402

SR = 16000


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    return soxr.resample(x, sr, SR) if sr != SR else x


def split_quiet(x: np.ndarray, chunk_s: float, search_s: float = 5.0) -> list[np.ndarray]:
    """Split into chunks of at most chunk_s, cutting at the lowest-energy 50 ms window in the last search_s."""
    chunks, start, n = [], 0, int(chunk_s * SR)
    win = int(0.05 * SR)
    while len(x) - start > n:
        lo, hi = start + n - int(search_s * SR), start + n
        seg = x[lo:hi]
        energy = np.convolve(seg**2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        chunks.append(x[start:cut])
        start = cut
    chunks.append(x[start:])
    return [c for c in chunks if len(c) > SR // 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunk_seconds", type=float, default=30.0)
    ap.add_argument("--batch_seconds", type=float, default=1200.0)
    ap.add_argument("audio", nargs="+")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    units = Units(ckpt["config"]["encoder_id"], ckpt["unit_gemma_ids"])
    model = GemmaCTC.from_checkpoint(ckpt).to(device).eval()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        model.float()

    for path in args.audio:
        x = load_audio(path)
        chunks = split_quiet(x, args.chunk_seconds)
        per_batch = max(1, int(args.batch_seconds // args.chunk_seconds))
        texts = []
        t0 = time.time()
        for i in range(0, len(chunks), per_batch):
            b = collate([(c * 32767).astype(np.int16) for c in chunks[i : i + per_batch]], None, [])
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits, lens = model(b["wav"].to(device), b["wav_lens"].to(device))
            texts += [units.decode(u) for u in greedy_decode(logits, lens)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"==> {path}  ({len(x) / SR:.1f}s audio, {dt:.2f}s, {len(x) / SR / dt:.0f}x realtime)")
        print(" ".join(t for t in texts if t))


if __name__ == "__main__":
    main()
