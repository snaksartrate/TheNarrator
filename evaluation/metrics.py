"""Word-level story metrics. Same normalisation for every model.

Rule: lowercase; a word = letters/digits, optionally joined by apostrophes
(don't = 1 word). Punctuation is NOT a word. Hyphenated forms split in two.
Metrics use normalised words, never tokenizer ids, so the three tokenizers
cannot redefine them.
"""
import re
from collections import Counter

REPETITION_N = 3
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def tokenize_words(text):
    return _WORD_RE.findall((text or "").lower().replace("\u2019", "'"))


def _ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def distinct_n(text, n):
    """unique n-grams / total n-grams; 0.0 if text has fewer than n words."""
    grams = _ngrams(tokenize_words(text), n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(text, n=REPETITION_N):
    """repeated n-gram occurrences / total n-grams (trigram by default), in [0, 1)."""
    grams = _ngrams(tokenize_words(text), n)
    if not grams:
        return 0.0
    return sum(c - 1 for c in Counter(grams).values()) / len(grams)


def lexical_diversity(text):
    """unique words / total words (length-sensitive: pair with word_count)."""
    words = tokenize_words(text)
    return len(set(words)) / len(words) if words else 0.0


def story_metrics(text):
    return {
        "word_count": len(tokenize_words(text)),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "repetition_rate": repetition_rate(text),
        "repetition_ngram_n": REPETITION_N,
        "lexical_diversity": lexical_diversity(text),
    }