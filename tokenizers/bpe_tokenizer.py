"""
Byte Pair Encoding (BPE) tokenizer written from scratch (one of the three tokenizers).

Idea:
  1. Start with every single character as a token.
  2. Find the most frequent pair of neighbouring tokens in the training text.
  3. Merge that pair into one new token and record the merge.
  4. Repeat until the vocabulary reaches `vocab_size`.

Common words end up as single tokens, rare words are built from pieces, and
nothing is ever unknown as long as the characters were seen in training.

Implementation notes:
  * Text is first cut into "pre tokens" (a word with its leading space, a
    punctuation mark, or a run of whitespace). Merges never cross these
    boundaries, and training runs on unique pre tokens weighted by frequency,
    which is far faster than scanning the raw text every iteration.
  * Pair counts are updated incrementally after each merge instead of being
    recomputed from scratch.
  * Encoding applies merges in the order they were learned (lowest rank first).
  * Decoding is lossless: just join the token strings.
"""

import re
from collections import Counter, defaultdict

from .base_tokenizer import BaseTokenizer

# " word" | " punct" | whitespace run
_PRETOKEN_RE = re.compile(r" ?\w+| ?[^\w\s]|\s+")


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size=2000, min_pair_freq=2):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq   # stop when the best pair is rarer than this
        self.merges = []                     # ordered list of (left, right)
        self.merge_ranks = {}                # (left, right) -> rank
        self._cache = {}                     # pre token -> list of symbols

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_pair(symbols, pair, new_symbol):
        """Replace every adjacent occurrence of `pair` in `symbols`."""
        out = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(new_symbol)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    def train(self, text):
        self._init_specials()
        self.merges = []
        self.merge_ranks = {}
        self._cache = {}

        # base vocabulary: every character in the training text
        for ch in sorted(set(text)):
            self._add_token(ch)

        # unique pre tokens with their frequency: [symbols, freq]
        pretoken_freq = Counter(_PRETOKEN_RE.findall(text))
        words = [[list(w), f] for w, f in sorted(pretoken_freq.items())]

        # initial pair counts and an index of which words contain each pair
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for idx, (symbols, freq) in enumerate(words):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += freq
                pair_words[pair].add(idx)

        while self.vocab_size < self.target_vocab_size and pair_counts:
            # most frequent pair, ties broken by the pair itself (deterministic)
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_count < self.min_pair_freq:
                break

            new_symbol = best[0] + best[1]
            self.merge_ranks[best] = len(self.merges)
            self.merges.append(best)
            self._add_token(new_symbol)

            # apply the merge only to the words that contain the pair
            for idx in list(pair_words[best]):
                symbols, freq = words[idx]
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] -= freq
                new_symbols = self._merge_pair(symbols, best, new_symbol)
                for pair in zip(new_symbols, new_symbols[1:]):
                    pair_counts[pair] += freq
                    pair_words[pair].add(idx)
                words[idx][0] = new_symbols

            del pair_counts[best]
            del pair_words[best]

    # ------------------------------------------------------------------
    def _bpe_word(self, pretoken):
        """Apply the learned merges to one pre token."""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        symbols = list(pretoken)
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if best not in self.merge_ranks:
                break
            symbols = self._merge_pair(symbols, best, best[0] + best[1])
        self._cache[pretoken] = symbols
        return symbols

    def _encode(self, text):
        unk = self.unk_id
        ids = []
        for pretoken in _PRETOKEN_RE.findall(text):
            for sym in self._bpe_word(pretoken):
                ids.append(self.token_to_id.get(sym, unk))
        return ids

    def _detokenize(self, tokens):
        return "".join(tokens)

    # ------------------------------------------------------------------
    def _get_extra_state(self):
        return {
            "target_vocab_size": self.target_vocab_size,
            "min_pair_freq": self.min_pair_freq,
            "merges": [list(p) for p in self.merges],
        }

    def _set_extra_state(self, state):
        self.target_vocab_size = state.get("target_vocab_size", self.target_vocab_size)
        self.min_pair_freq = state.get("min_pair_freq", self.min_pair_freq)
        self.merges = [tuple(p) for p in state.get("merges", [])]
        self.merge_ranks = {p: r for r, p in enumerate(self.merges)}
        self._cache = {}
