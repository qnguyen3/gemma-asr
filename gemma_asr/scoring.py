"""WER / CER / tone-insensitive WER, accumulated as counts so they can be summed across GPUs."""

from dataclasses import dataclass, fields

import jiwer
import torch
import torch.distributed as dist

from .text import normalize_hypothesis, strip_tones


@dataclass
class ErrorCounts:
    word_errors: int = 0
    ref_words: int = 0
    char_errors: int = 0
    ref_chars: int = 0
    word_errors_notone: int = 0
    utts: int = 0

    def __iadd__(self, other: "ErrorCounts") -> "ErrorCounts":
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def all_reduce(self, device) -> "ErrorCounts":
        if not (dist.is_available() and dist.is_initialized()):
            return self
        t = torch.tensor([getattr(self, f.name) for f in fields(self)], dtype=torch.long, device=device)
        dist.all_reduce(t)
        return ErrorCounts(*t.tolist())

    def summary(self) -> dict:
        return {
            "wer": self.word_errors / max(self.ref_words, 1),
            "cer": self.char_errors / max(self.ref_chars, 1),
            "wer_notone": self.word_errors_notone / max(self.ref_words, 1),
            "utts": self.utts,
            "ref_words": self.ref_words,
        }


def _errors(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    o = jiwer.process_words(refs, hyps)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def _char_errors(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    o = jiwer.process_characters(refs, hyps)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def score_batch(refs: list[str], hyps: list[str]) -> ErrorCounts:
    """refs are already normalized; hyps get the same normalization. Empty refs are skipped."""
    pairs = [(r, normalize_hypothesis(h)) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return ErrorCounts()
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    we, rw = _errors(refs, hyps)
    ce, rc = _char_errors(refs, hyps)
    we_nt, _ = _errors([strip_tones(r) for r in refs], [strip_tones(h) for h in hyps])
    return ErrorCounts(we, rw, ce, rc, we_nt, len(pairs))
