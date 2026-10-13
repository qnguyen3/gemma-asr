"""CTC output units = the subset of Gemma tokenizer tokens that occur in the Vietnamese training text.

Vietnamese is tokenized by Gemma at roughly one token per syllable, so this is essentially a syllable vocabulary
that lives in the embedding model's own text space (the head can be initialized from Gemma's token embeddings),
with sub-word fallback for rare syllables and foreign words.

Unit ids: 0 = CTC blank, 1 = <unk> (rare Gemma tokens), 2.. = kept Gemma tokens.
"""

import json
from collections import Counter

from transformers import AutoTokenizer

BLANK, UNK = 0, 1


class Units:
    def __init__(self, model_id: str, gemma_ids: list[int]):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.gemma_ids = list(gemma_ids)
        self.gemma_to_unit = {g: i + 2 for i, g in enumerate(self.gemma_ids)}

    def __len__(self) -> int:
        return len(self.gemma_ids) + 2

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        # Leading space so the first word gets the same "▁word" token as every other word.
        return self.tokenizer([" " + t for t in texts], add_special_tokens=False).input_ids

    def encode(self, texts: list[str]) -> list[list[int]]:
        return [[self.gemma_to_unit.get(g, UNK) for g in ids] for ids in self.tokenize(texts)]

    def decode(self, unit_ids: list[int]) -> str:
        gemma = [self.gemma_ids[u - 2] for u in unit_ids if u >= 2]
        return self.tokenizer.decode(gemma).strip()

    def save(self, path: str, extra: dict | None = None) -> None:
        with open(path, "w") as f:
            json.dump({"model_id": self.model_id, "gemma_ids": self.gemma_ids, **(extra or {})}, f)

    @classmethod
    def load(cls, path: str) -> "Units":
        with open(path) as f:
            d = json.load(f)
        return cls(d["model_id"], d["gemma_ids"])

    @classmethod
    def build(cls, model_id: str, texts, min_count: int = 2, batch_size: int = 10_000) -> tuple["Units", dict]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        counts: Counter = Counter()
        batch = []
        for t in texts:
            batch.append(" " + t)
            if len(batch) == batch_size:
                for ids in tokenizer(batch, add_special_tokens=False).input_ids:
                    counts.update(ids)
                batch = []
        if batch:
            for ids in tokenizer(batch, add_special_tokens=False).input_ids:
                counts.update(ids)
        kept = [g for g, c in counts.most_common() if c >= min_count]
        total = sum(counts.values())
        covered = sum(counts[g] for g in kept)
        stats = {
            "distinct_tokens": len(counts),
            "kept_tokens": len(kept),
            "total_tokens": total,
            "token_coverage": covered / max(total, 1),
        }
        return cls(model_id, kept), stats
