"""
Base class shared by all tokenizers in TheNarrator.

Every tokenizer (char, word, BPE) inherits from this class, so the model and
training code can use any of them through the same interface:

    tokenizer.train(text)
    ids = tokenizer.encode(text)
    text = tokenizer.decode(ids)
    n = tokenizer.vocab_size

The base class owns everything that is NOT specific to a tokenization method:
special tokens, the vocabulary dictionaries, saving/loading and statistics.
Subclasses only implement how text is split into tokens (train, _encode,
_detokenize).
"""

import json
from pathlib import Path


class BaseTokenizer:
    # Special tokens always occupy ids 0..3, in this order, for every tokenizer.
    PAD = "<pad>"   # padding for batching
    UNK = "<unk>"   # unknown token (not seen during training)
    BOS = "<bos>"   # beginning of story
    EOS = "<eos>"   # end of story
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._init_specials()

    # ------------------------------------------------------------------
    # vocabulary helpers
    # ------------------------------------------------------------------
    def _init_specials(self):
        """Reset the vocabulary so it contains only the special tokens."""
        self.token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def _add_token(self, token):
        """Add a token to the vocabulary if new and return its id."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def _check_trained(self):
        if self.vocab_size <= len(self.SPECIAL_TOKENS):
            raise RuntimeError(
                f"{type(self).__name__} has no vocabulary. "
                "Call train(text) or load(path) first."
            )

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self):
        return self.token_to_id[self.UNK]

    @property
    def bos_id(self):
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self):
        return self.token_to_id[self.EOS]

    # ------------------------------------------------------------------
    # public interface (mandatory for integration)
    # ------------------------------------------------------------------
    def train(self, text):
        """Build the vocabulary from raw text (training split only)."""
        raise NotImplementedError

    def encode(self, text, add_special_tokens=False):
        """Convert text to a list of integer ids.

        If add_special_tokens is True, BOS is prepended and EOS appended.
        """
        self._check_trained()
        ids = self._encode(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Convert a list of integer ids back to text."""
        self._check_trained()
        tokens = []
        for i in token_ids:
            i = int(i)  # also accepts tensors / numpy ints
            tok = self.id_to_token.get(i)
            if tok is None:
                continue  # id outside the vocabulary, ignore it
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return self._detokenize(tokens)

    # ------------------------------------------------------------------
    # methods subclasses must implement
    # ------------------------------------------------------------------
    def _encode(self, text):
        raise NotImplementedError

    def _detokenize(self, tokens):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # extra state (BPE merges, word min_freq, ...) stored next to the vocab
    # ------------------------------------------------------------------
    def _get_extra_state(self):
        return {}

    def _set_extra_state(self, state):
        pass

    # ------------------------------------------------------------------
    # saving and loading
    # ------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": type(self).__name__,
            "token_to_id": self.token_to_id,
            "extra": self._get_extra_state(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != type(self).__name__:
            raise ValueError(
                f"File holds a {data.get('type')}, "
                f"but you are loading it into {type(self).__name__}."
            )
        self.token_to_id = {tok: int(i) for tok, i in data["token_to_id"].items()}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self._set_extra_state(data.get("extra", {}))
        return self

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def get_stats(self, text):
        """Tokenization statistics for a piece of text."""
        ids = self.encode(text)
        n_tokens = len(ids)
        n_unk = sum(1 for i in ids if i == self.unk_id)
        return {
            "vocab_size": self.vocab_size,
            "num_chars": len(text),
            "num_tokens": n_tokens,
            "chars_per_token": len(text) / n_tokens if n_tokens else 0.0,
            "unk_count": n_unk,
            "unk_rate": n_unk / n_tokens if n_tokens else 0.0,
        }
