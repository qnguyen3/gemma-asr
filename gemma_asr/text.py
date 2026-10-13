"""Vietnamese transcript normalization.

Every transcript (train and eval) goes through `normalize_text` so that the model never has to learn
spelling variants that sound identical:

* Unicode NFC, lowercase, punctuation -> space, whitespace collapsed.
* Tone marks are moved to the "old style" position for the open syllables `oa`, `oe`, `uy`
  (hoà -> hòa, khoẻ -> khỏe, thuỷ -> thủy). This is the more common convention in Vietnamese text and
  the Gemma tokenizer has single tokens for these forms.
* Utterances containing digits or characters outside the Vietnamese/Latin alphabet are rejected
  (`None`), because we have no reliable way to know how numbers or foreign scripts were spoken.
"""

import re
import unicodedata
from functools import lru_cache

# Combining marks (NFD) used by Vietnamese.
TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}  # huyền, sắc, ngã, hỏi, nặng
QUALITY_MARKS = {"̂", "̆", "̛"}  # circumflex (â ê ô), breve (ă), horn (ơ ư)

_PUNCT_RE = re.compile(r"[^\w\s]|_")
_DIGIT_RE = re.compile(r"\d")


@lru_cache(maxsize=4096)
def _is_allowed_char(c: str) -> bool:
    if c == "đ" or ("a" <= c <= "z"):
        return True
    d = unicodedata.normalize("NFD", c)
    if not ("a" <= d[0] <= "z"):
        return False
    return all(m in TONE_MARKS or m in QUALITY_MARKS for m in d[1:])


@lru_cache(maxsize=200_000)
def canonicalize_tone(syllable: str) -> str:
    """Move the tone mark of open `oa`/`oe`/`uy` syllables onto the first vowel (old style)."""
    d = unicodedata.normalize("NFD", syllable)
    tones = [c for c in d if c in TONE_MARKS]
    if len(tones) != 1:
        return syllable
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONE_MARKS))
    for pair in ("oa", "oe", "uy"):
        if base.endswith(pair) and not (pair == "uy" and base.endswith("quy")):
            i = len(base) - 2
            toned = unicodedata.normalize("NFC", base[i] + tones[0])
            return base[:i] + toned + base[i + 1 :]
    return syllable


def strip_tones(text: str) -> str:
    """Remove tone marks but keep vowel quality (used to measure tone-only errors)."""
    d = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONE_MARKS))


def normalize_text(text: str, drop_digits: bool = True) -> str | None:
    """Normalize a transcript. Returns None if the utterance should be discarded."""
    if not text:
        return None
    s = unicodedata.normalize("NFC", text).lower()
    s = _PUNCT_RE.sub(" ", s)
    if drop_digits and _DIGIT_RE.search(s):
        return None
    words = s.split()
    if not words:
        return None
    for w in words:
        if not all(_is_allowed_char(c) for c in w):
            return None
    return " ".join(canonicalize_tone(w) for w in words)


def normalize_hypothesis(text: str) -> str:
    """Lenient normalization for model output / scoring (never rejects)."""
    s = unicodedata.normalize("NFC", text).lower()
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(canonicalize_tone(w) for w in s.split())
