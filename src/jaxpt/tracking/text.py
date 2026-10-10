"""Text diagnostics shared by evaluation and training: is an answer stuck in a repetition loop?

Calibrated on real Qwen3-0.6B-Base outputs (E0): a compression-ratio test did not separate loops from long,
on-track reasoning, so the rule looks for repeated sentences or a short phrase that keeps cycling.
"""
import re
from collections import Counter

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def is_degenerate(text, min_sentence=20, sentence_repeats=3, ngram=5, ngram_repeats=8, window_words=400):
    """True if a sentence (>= min_sentence chars) appears >= sentence_repeats times,
    or a ngram-word phrase appears >= ngram_repeats times in the last window_words words."""
    sentences = [s.strip().lower() for s in _SPLIT.split(text) if len(s.strip()) >= min_sentence]
    if sentences and Counter(sentences).most_common(1)[0][1] >= sentence_repeats:
        return True
    words = text.split()[-window_words:]
    grams = Counter(tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1))
    return bool(grams) and grams.most_common(1)[0][1] >= ngram_repeats
