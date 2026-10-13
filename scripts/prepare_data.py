#!/usr/bin/env python
"""Download Vietnamese speech datasets from the HF Hub and convert them to uniform training shards.

Output layout:
    <out_dir>/<split>/<source>/<shard>.parquet   columns: audio (FLAC 16k mono), text, duration, source
    <out_dir>/manifest.json                      hours / utterances per split and source

Every input parquet file becomes one output shard, so the conversion is parallel and resumable
(finished shards are skipped on re-run).

Example:
    python scripts/prepare_data.py --out_dir data/prepared --raw_dir data/raw --workers 96
"""

import argparse
import fnmatch
import glob
import io
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.text import normalize_text  # noqa: E402

# split name -> glob patterns inside the HF repo
SOURCES = {
    # ~500h, podcasts/books/travel..., all three regional accents. Gated (auto-approve): accept terms on HF.
    "bud500": dict(
        repo="linhtran92/viet_bud500", text="transcription",
        splits={"train": ["data/train-*"], "valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # ~1000h of YouTube speech (made for TTS, punctuated text). Gated (manual approval, can take days).
    "vivoice": dict(repo="capleaf/viVoice", text="text", splits={"train": ["data/train-*"]}),
    # VLSP 2020 (VinAI) ~100h
    "vlsp2020": dict(repo="doof-ferb/vlsp2020_vinai_100h", text="transcription", splits={"train": ["data/train-*"]}),
    # FPT Open Speech Data ~30h (mp3)
    "fpt_fosd": dict(repo="doof-ferb/fpt_fosd", text="transcription", splits={"train": ["data/train-*"]}),
    # InfoRe 1 (~25h single speaker) and InfoRe 2 audiobooks (~415h)
    "infore1": dict(repo="doof-ferb/infore1_25hours", text="transcription", splits={"train": ["data/train-*"]}),
    "infore2": dict(repo="doof-ferb/infore2_audiobooks", text="transcription", splits={"train": ["data/train-*"]}),
    # LSVSC ~100h, labelled dialect/topic
    "lsvsc": dict(
        repo="doof-ferb/LSVSC", text="transcription",
        splits={"train": ["data/train-*"], "valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # ViMD: 63 provincial dialects
    "vimd": dict(
        repo="nguyendv02/ViMD_Dataset", text="text",
        splits={"train": ["data/train-*"], "valid": ["data/valid-*"], "test": ["data/test-*"]},
    ),
    # Common Voice 17 (Vietnamese)
    "cv17": dict(
        repo="fixie-ai/common_voice_17_0", text="sentence",
        splits={"train": ["vi/train/*/*.parquet"], "valid": ["vi/validation/*/*.parquet"],
                "test": ["vi/test/*/*.parquet"]},
    ),
    # FLEURS vi_vn (read Wikipedia sentences)
    "fleurs": dict(
        repo="google/fleurs", text="raw_transcription",
        splits={"train": ["parquet-data/vi_vn/train-*"], "valid": ["parquet-data/vi_vn/validation-*"],
                "test": ["parquet-data/vi_vn/test-*"]},
    ),
    # GigaSpeech 2 Vietnamese dev/test (YouTube, hard). Eval only.
    "gigaspeech2": dict(
        repo="doof-ferb/gigaspeech2_vie", text="transcription",
        splits={"valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # PhoAudiobook ~940h. Gated with manual approval -> off by default, add with --sources ... phoaudiobook
    "phoaudiobook": dict(
        repo="thivux/phoaudiobook", text="text", default=False,
        splits={"train": ["data/train-*"], "valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
}
DEFAULT_SOURCES = [k for k, v in SOURCES.items() if v.get("default", True)]
SR = 16000
ROW_GROUP = 512


def download(source: str, splits: list[str], raw_dir: str, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    cfg = SOURCES[source]
    patterns = [p for s in splits for p in cfg["splits"].get(s, [])]
    if not patterns:
        return
    print(f"[download] {source}: {cfg['repo']} {patterns}", flush=True)
    snapshot_download(
        repo_id=cfg["repo"], repo_type="dataset", allow_patterns=patterns,
        local_dir=os.path.join(raw_dir, source), max_workers=max_workers,
    )


def local_files(source: str, split: str, raw_dir: str) -> list[str]:
    root = os.path.join(raw_dir, source)
    files = []
    for pat in SOURCES[source]["splits"].get(split, []):
        files += [f for f in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
                  if fnmatch.fnmatch(os.path.relpath(f, root), pat)]
    return sorted(set(files))


def to_mono16k(b: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ")
    return x


def encode_flac(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


SCHEMA = pa.schema([("audio", pa.binary()), ("text", pa.string()), ("duration", pa.float32()),
                    ("source", pa.string())])


def process_file(job: tuple) -> dict:
    source, split, in_path, out_path, text_col, min_dur, max_dur, delete_raw = job
    stats_path = out_path + ".stats.json"
    if os.path.exists(out_path) and os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    st = Counter()
    hours = 0.0
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="none")  # FLAC is already compressed
    rows = {k: [] for k in SCHEMA.names}

    def flush():
        if rows["audio"]:
            writer.write_table(pa.table(rows, schema=SCHEMA), row_group_size=ROW_GROUP)
            for v in rows.values():
                v.clear()

    pf = pq.ParquetFile(in_path)
    for batch in pf.iter_batches(batch_size=64, columns=["audio", text_col]):
        for audio, text in zip(batch.column("audio").to_pylist(), batch.column(text_col).to_pylist()):
            st["in"] += 1
            norm = normalize_text(text or "")
            if norm is None:
                st["skip_text"] += 1
                continue
            b = audio.get("bytes") if isinstance(audio, dict) else None
            if not b:
                st["skip_no_audio"] += 1
                continue
            try:
                x = to_mono16k(b)
            except Exception:
                st["skip_decode"] += 1
                continue
            dur = len(x) / SR
            if dur < min_dur or dur > max_dur:
                st["skip_duration"] += 1
                continue
            rows["audio"].append(encode_flac(x))
            rows["text"].append(norm)
            rows["duration"].append(dur)
            rows["source"].append(source)
            st["out"] += 1
            hours += dur / 3600
            if len(rows["audio"]) >= ROW_GROUP:
                flush()
    flush()
    writer.close()
    os.replace(tmp, out_path)
    result = {"source": source, "split": split, "file": out_path, "hours": hours, **st}
    with open(stats_path, "w") as f:
        json.dump(result, f)
    if delete_raw:
        os.remove(in_path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=list(SOURCES))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--download_workers", type=int, default=16)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=30.0)
    ap.add_argument("--max_files_per_source", type=int, default=0, help="debug: only convert N files per split")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--delete_raw", action="store_true", help="delete each raw parquet after converting it")
    args = ap.parse_args()

    if not args.skip_download:
        for s in args.sources:
            try:
                download(s, args.splits, args.raw_dir, args.download_workers)
            except Exception as e:  # gated / no access: keep going with the rest
                print(f"[download] !! {s} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)

    jobs = []
    for s in args.sources:
        cfg = SOURCES[s]
        for split in args.splits:
            files = local_files(s, split, args.raw_dir)
            if args.max_files_per_source:
                files = files[: args.max_files_per_source]
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                out = os.path.join(args.out_dir, split, s, f"{name}.parquet")
                jobs.append((s, split, f, out, cfg["text"], args.min_dur, args.max_dur, args.delete_raw))
    # big files first for better load balancing
    jobs.sort(key=lambda j: -os.path.getsize(j[2]) if os.path.exists(j[2]) else 0)
    print(f"[convert] {len(jobs)} files with {args.workers} workers", flush=True)

    results, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                h = sum(x["hours"] for x in results)
                print(f"[convert] {i}/{len(jobs)} files, {h:.1f}h kept, {time.time() - t0:.0f}s", flush=True)

    manifest = {}
    for r in results:
        m = manifest.setdefault(r["split"], {}).setdefault(r["source"], Counter())
        for k, v in r.items():
            if isinstance(v, (int, float)):
                m[k] += v
    # merge with an existing manifest (e.g. sources prepared in an earlier run)
    path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for split, d in old.items():
            for src, v in d.items():
                manifest.setdefault(split, {}).setdefault(src, v)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    print("\nsplit        source         hours     utts   dropped(text/dur/decode)")
    for split, d in sorted(manifest.items()):
        for src, v in sorted(d.items()):
            print(f"{split:12s} {src:12s} {v['hours']:8.1f} {v.get('out', 0):8d}   "
                  f"{v.get('skip_text', 0)}/{v.get('skip_duration', 0)}/{v.get('skip_decode', 0)}")
        print(f"{split:12s} {'TOTAL':12s} {sum(v['hours'] for v in d.values()):8.1f}")


if __name__ == "__main__":
    main()
