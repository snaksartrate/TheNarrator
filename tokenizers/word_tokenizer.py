"""
Word level tokenizer (one of the three tokenizers).

Text is split into words, punctuation marks and newlines. Each distinct one
seen at least `min_freq` times becomes a token. Anything rarer, or never
seen, is mapped to <unk>. Short sequences, but a large vocabulary and an
unknown word problem.

Decoding is not perfectly lossless: spaces are rebuilt with simple rules
(no space before . , ! ? etc, quotes are paired), so extra spaces or odd
spacing in the original text are not recovered.
"""

import re
from collections import Counter

from .base_tokenizer import BaseTokenizer

# a word (keeps contractions like don't together), or one punctuation mark, or a newline
_WORD_RE = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]|\n")

_NO_SPACE_BEFORE = set(".,!?;:)]}%\u2026\u201d")
_NO_SPACE_AFTER = set("([{\u201c")
_OPEN_QUOTE = "\x00open_quote"  # internal marker, never a real token


class WordTokenizer(BaseTokenizer):
    def __init__(self, min_freq=2, max_vocab=None, lowercase=False):
        super().__init__()
        self.min_freq = min_freq
        self.max_vocab = max_vocab      # cap on number of regular tokens, None = no cap
        self.lowercase = lowercase

    def _split(self, text):
        if self.lowercase:
            text = text.lower()
        return _WORD_RE.findall(text)

    def train(self, text):
        self._init_specials()
        counts = Counter(self._split(text))
        # most frequent first, ties broken alphabetically for determinism
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        added = 0
        for word, freq in ordered:
            if freq < self.min_freq:
                break
            if self.max_vocab is not None and added >= self.max_vocab:
                break
            self._add_token(word)
            added += 1

    def _encode(self, text):
        unk = self.unk_id
        return [self.token_to_id.get(w, unk) for w in self._split(text)]

    def _detokenize(self, tokens):
        out = []
        prev = None
        in_quote = False
        for tok in tokens:
            if tok == "\n":
                out.append("\n")
                prev = "\n"
                continue

            if tok == '"':
                if in_quote:                      # closing quote, glue to previous
                    out.append(tok)
                    in_quote = False
                    prev = tok
                else:                             # opening quote
                    if prev is not None and prev != "\n" and prev not in _NO_SPACE_AFTER:
                        out.append(" ")
                    out.append(tok)
                    in_quote = True
                    prev = _OPEN_QUOTE
                continue

            if (
                prev is None
                or prev == "\n"
                or prev == _OPEN_QUOTE
                or prev in _NO_SPACE_AFTER
                or tok in _NO_SPACE_BEFORE
            ):
                space = ""
            else:
                space = " "
            out.append(space + tok)
            prev = tok
        return "".join(out)

    def _get_extra_state(self):
        return {
            "min_freq": self.min_freq,
            "max_vocab": self.max_vocab,
            "lowercase": self.lowercase,
        }

    def _set_extra_state(self, state):
        self.min_freq = state.get("min_freq", self.min_freq)
        self.max_vocab = state.get("max_vocab", self.max_vocab)
        self.lowercase = state.get("lowercase", self.lowercase)
