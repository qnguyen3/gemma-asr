#!/usr/bin/env python
"""Compare a gemma-asr checkpoint with HF ASR models (Whisper, PhoWhisper, Cohere Transcribe) on the same
test utterances, with the same normalization and scoring. Reports WER / CER / tone-insensitive WER and speed.

    python scripts/benchmark.py --model gemma:runs/vi/best.pt --out results/gemma.json
    python scripts/benchmark.py --model whisper:openai/whisper-large-v3 --out results/whisper-large-v3.json
    python scripts/benchmark.py --model cohere:CohereLabs/cohere-transcribe-03-2026 --out results/cohere.json

Speed is audio seconds / wall-clock seconds from waveform to text (feature extraction + model + decoding),
excluding FLAC decoding and one warmup batch. Run one model per GPU.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.data import EvalSet, decode_flac, list_shards  # noqa: E402
from gemma_asr.scoring import ErrorCounts, score_batch  # noqa: E402

SR = 16000

# ---------------------------------------------------------------- Vietnamese number reading for hypotheses
# References never contain digits (those utterances were dropped), but Whisper-style models write "42".
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_under_1000(n: int, full: bool) -> list[str]:
    h, t, u = n // 100, n // 10 % 10, n % 10
    out = []
    if h or full:
        out += [_DIGITS[h], "trăm"]
    if t == 0:
        if u:
            if out:
                out.append("lẻ")
            out.append(_DIGITS[u])
    elif t == 1:
        out.append("mười")
        if u:
            out.append("lăm" if u == 5 else _DIGITS[u])
    else:
        out += [_DIGITS[t], "mươi"]
        if u:
            out.append({1: "mốt", 5: "lăm"}.get(u, _DIGITS[u]))
    return out


def read_number(n: int) -> str:
    if n == 0:
        return "không"
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ"]
    out = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        out += _read_under_1000(g, full=bool(out))
        if scales[i]:
            out.append(scales[i])
    return " ".join(out)


def verbalize_digits(text: str) -> str:
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)  # 1.000 / 1,000 -> 1000
    text = re.sub(r"(\d+)\s*%", r"\1 phần trăm", text)
    return re.sub(r"\d+", lambda m: f" {read_number(int(m.group()))} " if len(m.group()) <= 15 else m.group(), text)


# ---------------------------------------------------------------- model runners
class GemmaRunner:
    def __init__(self, path, device):
        from gemma_asr.model import GemmaCTC, greedy_decode
        from gemma_asr.units import Units

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.units = Units(ckpt["config"]["encoder_id"], ckpt["unit_gemma_ids"])
        self.model = GemmaCTC.from_checkpoint(ckpt).to(device).eval()
        self.greedy_decode, self.device = greedy_decode, device

    def __call__(self, wavs):
        lens = torch.tensor([len(w) for w in wavs])
        wav = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            wav[i, : len(w)] = torch.from_numpy(w)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, out_lens = self.model(wav.to(self.device), lens.to(self.device))
        return [self.units.decode(u) for u in self.greedy_decode(logits, out_lens)]


class WhisperRunner:
    def __init__(self, model_id, device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
        self.device = device

    def __call__(self, wavs):
        feats = self.processor.feature_extractor(
            wavs, sampling_rate=SR, return_tensors="pt", device=str(self.device)).input_features
        with torch.no_grad():
            ids = self.model.generate(feats.to(self.device, torch.float16), language="vi", task="transcribe",
                                      num_beams=1, do_sample=False)
        return self.processor.batch_decode(ids, skip_special_tokens=True)


class CohereRunner:
    def __init__(self, model_id, device):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def __call__(self, wavs):
        inputs = self.processor(wavs, sampling_rate=SR, return_tensors="pt", language="vi", punctuation=False)
        chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.device, dtype=self.model.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
        return self.processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunk_index, language="vi")


RUNNERS = {"gemma": GemmaRunner, "whisper": WhisperRunner, "cohere": CohereRunner}


def batches_by_count(rows, n):
    for i in range(0, len(rows), n):
        yield rows[i: i + n]


def batches_by_seconds(rows, max_s):
    cur = []
    for r in rows:
        if cur and r[2] * (len(cur) + 1) > max_s:
            yield cur
            cur = []
        cur.append(r)
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gemma:<ckpt> | whisper:<hf id> | cohere:<hf id>")
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_utts", type=int, default=700, help="per test set, fixed random sample")
    ap.add_argument("--batch_size", type=int, default=32, help="HF models (utterances per batch)")
    ap.add_argument("--max_batch_seconds", type=float, default=1200.0, help="gemma model (padded audio per batch)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kind, ident = args.model.split(":", 1)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    runner = RUNNERS[kind](ident, device)

    sets = {name: EvalSet(files, args.max_utts) for name, files in list_shards(args.data_dir, args.split).items()}
    # decode FLAC up front so it isn't timed
    data = {name: [(decode_flac(a).astype(np.float32) / 32768.0, t, d) for a, t, d in es.rows]
            for name, es in sets.items()}

    def make_batches(rows):
        if kind == "gemma":
            return list(batches_by_seconds(rows, args.max_batch_seconds))
        return list(batches_by_count(rows, args.batch_size))

    # warmup on the first batch of the first set (untimed)
    first = make_batches(next(iter(data.values())))[0]
    runner([r[0] for r in first])
    torch.cuda.synchronize()

    results, dump = {}, []
    total_audio, total_time = 0.0, 0.0
    for name, rows in data.items():
        counts, audio_s, compute_s, digit_hyps = ErrorCounts(), 0.0, 0.0, 0
        for b in make_batches(rows):
            wavs = [r[0] for r in b]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hyps = runner(wavs)
            torch.cuda.synchronize()
            compute_s += time.perf_counter() - t0
            audio_s += sum(len(w) for w in wavs) / SR
            raw = list(hyps)
            digit_hyps += sum(bool(re.search(r"\d", h)) for h in hyps)
            hyps = [verbalize_digits(h) for h in hyps]
            refs = [r[1] for r in b]
            counts += score_batch(refs, hyps)
            dump += [{"set": name, "ref": r, "hyp": h, "raw": x} for r, h, x in zip(refs, hyps, raw)]
        s = counts.summary()
        s.update(hours=audio_s / 3600, speed=audio_s / compute_s, hyps_with_digits=digit_hyps)
        results[name] = s
        total_audio += audio_s
        total_time += compute_s
        print(f"{name:14s} WER {s['wer']:7.2%}  CER {s['cer']:7.2%}  WER-notone {s['wer_notone']:7.2%}  "
              f"{s['utts']:5d} utts  {s['hours']:5.2f}h  {s['speed']:7.0f}x RT", flush=True)

    summary = {
        "model": args.model,
        "avg_wer": float(np.mean([s["wer"] for s in results.values()])),
        "speed": total_audio / total_time,
        "params": sum(p.numel() for p in getattr(runner, "model").parameters()),
        "sets": results,
    }
    print(f"AVERAGE WER {summary['avg_wer']:.2%}  overall speed {summary['speed']:.0f}x RT  "
          f"params {summary['params'] / 1e6:.0f}M", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.splitext(args.out)[0] + ".hyps.jsonl", "w") as f:
        for d in dump:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
