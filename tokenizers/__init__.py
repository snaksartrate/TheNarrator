"""
Tokenizers package for TheNarrator.

Usage:
    from tokenizers import CharTokenizer, WordTokenizer, BPETokenizer
    from tokenizers import get_tokenizer, load_tokenizer

    tok = get_tokenizer("bpe", vocab_size=4000)
    tok.train(open("dataset/processed/train.txt", encoding="utf-8").read())
    ids = tok.encode("Once upon a time", add_special_tokens=True)
    text = tok.decode(ids)
"""

import json

from .base_tokenizer import BaseTokenizer
from .char_tokenizer import CharTokenizer
from .word_tokenizer import WordTokenizer
from .bpe_tokenizer import BPETokenizer

TOKENIZERS = {
    "char": CharTokenizer,
    "word": WordTokenizer,
    "bpe": BPETokenizer,
}


def get_tokenizer(name, **kwargs):
    """Create an untrained tokenizer by name: 'char', 'word' or 'bpe'."""
    try:
        return TOKENIZERS[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown tokenizer '{name}'. Choose from {list(TOKENIZERS)}")


def load_tokenizer(path):
    """Load any saved tokenizer file, the right class is picked automatically."""
    with open(path, encoding="utf-8") as f:
        kind = json.load(f)["type"]
    for cls in TOKENIZERS.values():
        if cls.__name__ == kind:
            return cls().load(path)
    raise ValueError(f"Unknown tokenizer type in file: {kind}")


__all__ = [
    "BaseTokenizer",
    "CharTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "TOKENIZERS",
    "get_tokenizer",
    "load_tokenizer",
]
