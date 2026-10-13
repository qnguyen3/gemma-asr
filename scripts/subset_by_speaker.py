#!/usr/bin/env python
"""Write a speaker-stratified fraction of a prepared training source as a new source.

The prepared shards don't keep speaker ids, so each prepared row is matched back to its raw row (prepared shards
keep the raw row order and only drop rows, so an in-order match on normalized text recovers the speaker).
Then every speaker keeps `--fraction` of their utterances (at least one), sampled with a fixed seed.

    python scripts/subset_by_speaker.py --source phoaudiobook --fraction 0.4 \
        --raw_dir /audio_data/raw --data_dir data/prepared --out_name phoaudiobook_40
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.text import normalize_text  # noqa: E402

ROW_GROUP = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="phoaudiobook")
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--raw_dir", default="/audio_data/raw")
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--out_name", default=None, help="default: <source>_<pct>")
    ap.add_argument("--speaker_col", default="speaker")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_name = args.out_name or f"{args.source}_{round(args.fraction * 100)}"
    out_dir = os.path.join(args.data_dir, "train", out_name)

    # 1. recover the speaker of every prepared row
    rows = []  # (prepared file, row index, speaker, duration)
    prepared = sorted(glob.glob(os.path.join(args.data_dir, "train", args.source, "*.parquet")))
    for pf in prepared:
        raw = glob.glob(os.path.join(args.raw_dir, args.source, "**", os.path.basename(pf)), recursive=True)
        assert len(raw) == 1, f"raw file for {pf}: {raw}"
        r = pq.read_table(raw[0], columns=[args.speaker_col, args.text_col])
        speakers, texts = r.column(args.speaker_col).to_pylist(), r.column(args.text_col).to_pylist()
        p = pq.read_table(pf, columns=["text", "duration"])
        j = 0
        for i, (text, dur) in enumerate(zip(p.column("text").to_pylist(), p.column("duration").to_pylist())):
            while j < len(texts) and normalize_text(texts[j] or "") != text:
                j += 1
            assert j < len(texts), f"could not match row {i} of {pf}"
            rows.append((pf, i, speakers[j], dur))
            j += 1

    # 2. stratified sample per speaker
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row[2]].append(row)
    rng = random.Random(args.seed)
    keep = defaultdict(set)
    per_speaker = {}
    for spk in sorted(by_speaker):
        items = by_speaker[spk]
        rng.shuffle(items)
        n = max(1, round(len(items) * args.fraction))
        for pf, i, _, _ in items[:n]:
            keep[pf].add(i)
        per_speaker[spk] = {"total": len(items), "kept": n}

    # 3. write the subset with the same schema
    os.makedirs(out_dir, exist_ok=True)
    kept_rows, kept_hours = 0, 0.0
    for pf in prepared:
        idx = sorted(keep.get(pf, ()))
        if not idx:
            continue
        t = pq.read_table(pf).take(idx)
        pq.write_table(t, os.path.join(out_dir, os.path.basename(pf)), row_group_size=ROW_GROUP, compression="none")
        kept_rows += t.num_rows
        kept_hours += sum(t.column("duration").to_pylist()) / 3600

    total_hours = sum(r[3] for r in rows) / 3600
    stats = {
        "source": args.source, "fraction": args.fraction, "seed": args.seed,
        "speakers": len(by_speaker), "utts_total": len(rows), "utts_kept": kept_rows,
        "hours_total": total_hours, "hours_kept": kept_hours, "per_speaker": per_speaker,
    }
    with open(os.path.join(args.data_dir, f"{out_name}.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"{out_name}: {len(by_speaker)} speakers, {kept_rows}/{len(rows)} utts, "
          f"{kept_hours:.1f}/{total_hours:.1f} h -> {out_dir}")


if __name__ == "__main__":
    main()
