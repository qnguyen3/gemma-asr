"""Data loading for prepared shards.

Prepared shards (written by scripts/prepare_data.py) are parquet files with columns
    audio: FLAC bytes (16 kHz mono, PCM_16)    text: normalized transcript
    duration: seconds (float32)                source: dataset name

Training uses an infinite, sharded, shuffled stream of *row groups* with duration-bucketed dynamic batches
(`max_batch_seconds` of padded audio per GPU). Training is step-based, so DDP ranks never disagree about
epoch boundaries.
"""

import glob
import io
import os
import random
import zlib

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .units import Units

FRAME_RATE = 25  # encoder output frames per second (40 ms)


def list_shards(data_dir: str, split: str, sources: list[str] | None = None) -> dict[str, list[str]]:
    """Returns {source: [parquet files]} for data_dir/<split>/<source>/*.parquet."""
    out = {}
    for src_dir in sorted(glob.glob(os.path.join(data_dir, split, "*"))):
        name = os.path.basename(src_dir)
        if sources and name not in sources:
            continue
        files = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
        if files:
            out[name] = files
    return out


def decode_flac(b: bytes) -> np.ndarray:
    x, _ = sf.read(io.BytesIO(b), dtype="int16")
    return x


def collate(wavs: list[np.ndarray], targets: list[list[int]] | None, texts: list[str]) -> dict:
    lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    wav = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
    for i, w in enumerate(wavs):
        wav[i, : len(w)] = torch.from_numpy(w.astype(np.float32) / 32768.0)
    batch = {"wav": wav, "wav_lens": lens, "texts": texts}
    if targets is not None:
        batch["targets"] = torch.tensor([t for ts in targets for t in ts], dtype=torch.long)
        batch["target_lens"] = torch.tensor([len(ts) for ts in targets], dtype=torch.long)
    return batch


class TrainStream(IterableDataset):
    def __init__(
        self,
        files: list[str],
        units_path: str,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        max_batch_seconds: float = 1200.0,
        max_batch_size: int = 512,
        buffer_utts: int = 4000,
        source_weights: dict[str, float] | None = None,
    ):
        self.files = files
        self.units_path = units_path
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.max_batch_seconds = max_batch_seconds
        self.max_batch_size = max_batch_size
        self.buffer_utts = buffer_utts
        # (file, row_group) work items; optional per-source repetition for upsampling small sets
        self.items = []
        for f in files:
            src = os.path.basename(os.path.dirname(f))
            reps = (source_weights or {}).get(src, 1.0)
            n_rg = pq.ParquetFile(f).metadata.num_row_groups
            whole, frac = int(reps), reps - int(reps)
            for rg in range(n_rg):
                self.items += [(f, rg)] * whole
                if frac > 0 and random.Random(zlib.crc32(f"{f}:{rg}:{seed}".encode())).random() < frac:
                    self.items.append((f, rg))

    def _batches_from_buffer(self, buf: list, rng: random.Random):
        buf.sort(key=lambda r: r[2])
        batches, cur, cur_max = [], [], 0.0
        for r in buf:
            m = max(cur_max, r[2])
            if cur and (m * (len(cur) + 1) > self.max_batch_seconds or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, m = [], r[2]
            cur.append(r)
            cur_max = m
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def _make_batch(self, rows: list, units: Units) -> dict | None:
        texts = [r[1] for r in rows]
        targets = units.encode(texts)
        wavs, keep_t, keep_txt = [], [], []
        for (audio, text, dur), tgt in zip(rows, targets):
            # CTC needs at least one frame per label (+ separating blanks for repeats)
            if not tgt or len(tgt) > dur * FRAME_RATE - 2:
                continue
            wavs.append(decode_flac(audio))
            keep_t.append(tgt)
            keep_txt.append(text)
        if not wavs:
            return None
        return collate(wavs, keep_t, keep_txt)

    def __iter__(self):
        units = Units.load(self.units_path)
        wi = get_worker_info()
        n_workers, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        gid, total = self.rank * n_workers + wid, self.world_size * n_workers
        epoch = 0
        while True:
            rng = random.Random(self.seed * 1000003 + epoch * 7919 + gid)
            order = list(range(len(self.items)))
            random.Random(self.seed + epoch).shuffle(order)  # same global order on every worker
            mine = [self.items[i] for i in order[gid::total]]
            buf = []
            for f, rg in mine:
                t = pq.ParquetFile(f).read_row_group(rg, columns=["audio", "text", "duration"])
                rows = list(zip(t.column("audio").to_pylist(), t.column("text").to_pylist(),
                                t.column("duration").to_pylist()))
                buf.extend(rows)
                if len(buf) >= self.buffer_utts:
                    rng.shuffle(buf)
                    head, buf = buf[: self.buffer_utts], buf[self.buffer_utts :]
                    for b in self._batches_from_buffer(head, rng):
                        batch = self._make_batch(b, units)
                        if batch is not None:
                            yield batch
            for b in self._batches_from_buffer(buf, rng):
                batch = self._make_batch(b, units)
                if batch is not None:
                    yield batch
            epoch += 1


class EvalSet:
    """A fixed, length-sorted list of utterances, sharded across ranks."""

    def __init__(self, files: list[str], max_utts: int | None = None, seed: int = 0):
        rows = []
        for f in files:
            t = pq.read_table(f, columns=["audio", "text", "duration"])
            rows += list(zip(t.column("audio").to_pylist(), t.column("text").to_pylist(),
                             t.column("duration").to_pylist()))
        if max_utts and len(rows) > max_utts:
            random.Random(seed).shuffle(rows)
            rows = rows[:max_utts]
        self.rows = sorted(rows, key=lambda r: r[2])
        self.hours = sum(r[2] for r in self.rows) / 3600

    def batches(self, rank: int = 0, world_size: int = 1, max_batch_seconds: float = 1200.0):
        mine = self.rows[rank::world_size]
        cur = []
        for r in mine:
            if cur and r[2] * (len(cur) + 1) > max_batch_seconds:
                yield collate([decode_flac(x[0]) for x in cur], None, [x[1] for x in cur])
                cur = []
            cur.append(r)
        if cur:
            yield collate([decode_flac(x[0]) for x in cur], None, [x[1] for x in cur])
