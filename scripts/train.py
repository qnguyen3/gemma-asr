#!/usr/bin/env python
"""Train the CTC reader on top of the frozen EmbeddingGemma 2 audio encoder.

Single GPU:   python scripts/train.py --data_dir data/prepared --out_dir runs/vi
8 GPUs:       torchrun --nproc_per_node 8 scripts/train.py --data_dir data/prepared --out_dir runs/vi

`--max_batch_seconds` is padded audio per GPU per step; the global batch is that x world size.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.data import EvalSet, TrainStream, list_shards  # noqa: E402
from gemma_asr.model import GemmaCTC, ModelConfig, greedy_decode, load_gemma_parts  # noqa: E402
from gemma_asr.scoring import ErrorCounts, score_batch  # noqa: E402
from gemma_asr.units import Units  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--units", default=None, help="default: <data_dir>/units.json")
    ap.add_argument("--out_dir", default="runs/vi")
    ap.add_argument("--train_sources", nargs="*", default=None, help="default: all sources in <data_dir>/train")
    ap.add_argument("--source_weights", default="{}", help='JSON, e.g. \'{"fleurs": 3, "vivoice": 0.5}\'')
    ap.add_argument("--eval_split", default="valid")
    ap.add_argument("--max_eval_utts", type=int, default=1500, help="per eval source")
    # model
    ap.add_argument("--encoder_id", default="google/embeddinggemma-2")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_layer_mix", action="store_true", help="use only the last encoder layer")
    ap.add_argument("--unfreeze_top_k", type=int, default=0, help="fine-tune the top K encoder layers")
    ap.add_argument("--head_init", choices=["text", "random"], default="text")
    ap.add_argument("--freeze_head", action="store_true")
    ap.add_argument("--no_specaug", action="store_true")
    # optimization
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--max_batch_seconds", type=float, default=1200.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=5e-5, help="lr for unfrozen encoder layers")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default=None, help="path to a checkpoint (e.g. runs/vi/last.pt)")
    ap.add_argument("--wandb", default=None, help="wandb project name (optional)")
    return ap.parse_args()


def setup_dist():
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", device_id=device)
        return dist.get_rank(), dist.get_world_size(), device
    return 0, 1, device


def lr_lambda(step, warmup, max_steps, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, units, eval_sets, rank, world, device, max_batch_seconds):
    model.eval()
    results = {}
    for name, es in eval_sets.items():
        counts, example = ErrorCounts(), None
        for batch in es.batches(rank, world, max_batch_seconds):
            wav, lens = batch["wav"].to(device, non_blocking=True), batch["wav_lens"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, out_lens = model(wav, lens)
            hyps = [units.decode(u) for u in greedy_decode(logits, out_lens)]
            counts += score_batch(batch["texts"], hyps)
            example = (batch["texts"][-1], hyps[-1])
        counts = counts.all_reduce(device)
        results[name] = counts.summary()
        if rank == 0 and counts.ref_words:
            print(f"  [eval] {name:12s} WER {results[name]['wer']:6.2%}  CER {results[name]['cer']:6.2%}  "
                  f"WER(no tone) {results[name]['wer_notone']:6.2%}  ({es.hours:.1f}h)", flush=True)
            if example:
                print(f"         ref: {example[0]}\n         hyp: {example[1]}", flush=True)
    model.train()
    if results:
        results["avg_wer"] = sum(r["wer"] for r in results.values()) / len(results)
    return results


def main():
    args = parse_args()
    rank, world, device = setup_dist()
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    units_path = args.units or os.path.join(args.data_dir, "units.json")
    units = Units.load(units_path)
    log(f"units: {len(units)}  world size: {world}")

    # ---- model (rank 0 downloads first)
    if rank != 0 and dist.is_initialized():
        dist.barrier()
    encoder, unit_emb = load_gemma_parts(args.encoder_id, units.gemma_ids)
    if rank == 0 and dist.is_initialized():
        dist.barrier()
    cfg = ModelConfig(
        encoder_id=args.encoder_id, vocab_size=len(units), d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, dropout=args.dropout, layer_mix=not args.no_layer_mix,
        unfreeze_top_k=args.unfreeze_top_k, head_init=args.head_init, freeze_head=args.freeze_head,
    )
    model = GemmaCTC(cfg, encoder, unit_emb).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {n_train / 1e6:.1f}M  (encoder frozen except top {cfg.unfreeze_top_k} layers)")

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    decay = [p for p in other if p.ndim >= 2]
    no_decay = [p for p in other if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ]
    if enc_params:
        groups.append({"params": enc_params, "weight_decay": args.weight_decay, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    step = 0
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best = ckpt["step"], ckpt.get("best", best)
        log(f"resumed from {args.resume} at step {step}")

    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False) if dist.is_initialized() else model

    # ---- data
    train_files = [f for fs in list_shards(args.data_dir, "train", args.train_sources).values() for f in fs]
    assert train_files, "no training shards found"
    stream = TrainStream(
        train_files, units_path, rank=rank, world_size=world, seed=args.seed + step,
        max_batch_seconds=args.max_batch_seconds, source_weights=json.loads(args.source_weights),
    )
    loader = DataLoader(stream, batch_size=None, num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=4 if args.num_workers else None, persistent_workers=args.num_workers > 0)
    eval_sets = {
        name: EvalSet(files, args.max_eval_utts, seed=args.seed)
        for name, files in list_shards(args.data_dir, args.eval_split).items()
    }
    log(f"train shards: {len(train_files)}  eval sets: {', '.join(f'{k} ({v.hours:.1f}h)' for k, v in eval_sets.items())}")

    wb = None
    if args.wandb and rank == 0:
        import wandb

        wb = wandb.init(project=args.wandb, config={**vars(args), "model": asdict(cfg)})

    def save(name, extra=None):
        if rank != 0:
            return
        ckpt = {
            "model": model.trainable_state_dict(), "config": model.config_dict(), "units": units_path,
            "unit_gemma_ids": units.gemma_ids, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "step": step, "best": best, "args": vars(args), **(extra or {}),
        }
        tmp = os.path.join(args.out_dir, name + ".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, os.path.join(args.out_dir, name))

    # ---- train
    model.train()
    metrics_f = open(os.path.join(args.out_dir, "metrics.jsonl"), "a") if rank == 0 else None
    t_last, audio_acc, loss_acc, n_acc, skipped = time.time(), 0.0, 0.0, 0, 0
    data_iter = iter(loader)
    while step < args.max_steps:
        batch = next(data_iter)
        wav, lens = batch["wav"].to(device, non_blocking=True), batch["wav_lens"].to(device, non_blocking=True)
        targets, tlens = batch["targets"].to(device, non_blocking=True), batch["target_lens"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, out_lens = ddp(wav, lens, specaug=not args.no_specaug)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T, B, V)
        loss_sum = F.ctc_loss(log_probs, targets, out_lens, tlens, blank=0, reduction="sum", zero_infinity=True)
        loss = loss_sum / tlens.sum().clamp(min=1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            skipped += 1
        sched.step()
        step += 1

        audio_acc += float(lens.sum()) / 16000
        loss_acc += loss.item()
        n_acc += 1
        if step % args.log_every == 0:
            dt = time.time() - t_last
            stats = torch.tensor([audio_acc, loss_acc / n_acc], device=device)
            if dist.is_initialized():
                dist.all_reduce(stats)
            audio_all, loss_avg = stats[0].item(), stats[1].item() / world
            rec = {
                "step": step, "loss": loss_avg, "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm),
                "audio_hours_per_s": audio_all / 3600 / dt, "rtf_x": audio_all / dt, "s_per_step": dt / n_acc,
                "skipped": skipped, "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            if model.layer_logits is not None and step % (args.log_every * 10) == 0:
                rec["layer_weights"] = [round(x, 3) for x in torch.softmax(model.layer_logits.float(), 0).tolist()]
            log(
                f"step {step:6d}  loss {loss_avg:.3f}  lr {rec['lr']:.2e}  gnorm {rec['grad_norm']:.2f}  "
                f"{rec['s_per_step']:.2f}s/step  {rec['audio_hours_per_s'] * 3600:.0f}x realtime  "
                f"mem {rec['gpu_mem_gb']:.1f}GB" + (f"\n  layer weights {rec['layer_weights']}" if "layer_weights" in rec else "")
            )
            if metrics_f:
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
            if wb:
                wb.log({k: v for k, v in rec.items() if k != "layer_weights"}, step=step)
            t_last, audio_acc, loss_acc, n_acc = time.time(), 0.0, 0.0, 0

        if step % args.eval_every == 0 or step == args.max_steps:
            log(f"[eval @ step {step}]")
            res = evaluate(model, units, eval_sets, rank, world, device, args.max_batch_seconds)
            if rank == 0 and res:
                metrics_f.write(json.dumps({"step": step, "eval": res}) + "\n")
                metrics_f.flush()
                if wb:
                    wb.log({f"eval/{k}/wer": v["wer"] for k, v in res.items() if isinstance(v, dict)}, step=step)
                if res["avg_wer"] < best:
                    best = res["avg_wer"]
                    save("best.pt", {"eval": res})
                    log(f"  new best avg WER {best:.2%} -> best.pt")
        if step % args.save_every == 0 or step == args.max_steps:
            save("last.pt")

    log("done.")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
