#!/usr/bin/env python
"""Build the CTC unit inventory (a subset of Gemma tokens) from the prepared training transcripts.

    python scripts/build_vocab.py --data_dir data/prepared --out data/prepared/units.json
"""

import argparse
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.data import list_shards  # noqa: E402
from gemma_asr.units import Units  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model_id", default="google/embeddinggemma-2")
    ap.add_argument("--min_count", type=int, default=2, help="rarer Gemma tokens map to <unk>")
    args = ap.parse_args()
    out = args.out or os.path.join(args.data_dir, "units.json")

    shards = list_shards(args.data_dir, "train")
    assert shards, f"no training shards under {args.data_dir}/train"

    def texts():
        for files in shards.values():
            for f in files:
                yield from pq.read_table(f, columns=["text"]).column("text").to_pylist()

    units, stats = Units.build(args.model_id, texts(), min_count=args.min_count)
    units.save(out, extra={"stats": stats})
    print(f"wrote {out}: {len(units)} units (incl. blank+unk)")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  most frequent:", units.tokenizer.convert_ids_to_tokens(units.gemma_ids[:30]))


if __name__ == "__main__":
    main()
