#!/usr/bin/env python
"""Evaluate a checkpoint on prepared eval shards (WER, CER, tone-insensitive WER, speed).

    python scripts/evaluate.py --ckpt runs/vi/best.pt --data_dir data/prepared --split test
    torchrun --nproc_per_node 8 scripts/evaluate.py --ckpt runs/vi/best.pt --split test
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.data import EvalSet, list_shards  # noqa: E402
from gemma_asr.model import GemmaCTC, greedy_decode  # noqa: E402
from gemma_asr.scoring import ErrorCounts, score_batch  # noqa: E402
from gemma_asr.units import Units  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--max_utts", type=int, default=0, help="0 = all")
    ap.add_argument("--max_batch_seconds", type=float, default=1200.0)
    ap.add_argument("--dump", default=None, help="write ref/hyp pairs to this jsonl (rank 0 part only)")
    args = ap.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    units = Units(ckpt["config"]["encoder_id"], ckpt["unit_gemma_ids"])
    model = GemmaCTC.from_checkpoint(ckpt).to(device).eval()
    dump = open(args.dump, "w") if args.dump and rank == 0 else None

    rows = []
    for name, files in list_shards(args.data_dir, args.split, args.sources).items():
        es = EvalSet(files, args.max_utts or None)
        counts, audio_s, compute_s = ErrorCounts(), 0.0, 0.0
        for batch in es.batches(rank, world, args.max_batch_seconds):
            wav, lens = batch["wav"].to(device), batch["wav_lens"].to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, out_lens = model(wav, lens)
            hyps = [units.decode(u) for u in greedy_decode(logits, out_lens)]
            compute_s += time.time() - t0
            audio_s += float(lens.sum()) / 16000
            counts += score_batch(batch["texts"], hyps)
            if dump:
                for r, h in zip(batch["texts"], hyps):
                    dump.write(json.dumps({"set": name, "ref": r, "hyp": h}, ensure_ascii=False) + "\n")
        counts = counts.all_reduce(device)
        s = counts.summary()
        rows.append((name, es.hours, s, audio_s / max(compute_s, 1e-9)))

    if rank == 0:
        print(f"\n{'set':14s} {'hours':>6s} {'utts':>7s} {'WER':>7s} {'CER':>7s} {'WER-notone':>10s} {'speed (x RT, rank0)':>20s}")
        for name, hours, s, speed in rows:
            print(f"{name:14s} {hours:6.1f} {s['utts']:7d} {s['wer']:7.2%} {s['cer']:7.2%} {s['wer_notone']:10.2%} {speed:20.0f}")
        if rows:
            print(f"{'AVERAGE':14s} {'':6s} {'':7s} {sum(r[2]['wer'] for r in rows) / len(rows):7.2%}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
