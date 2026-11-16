"""
Character level tokenizer (one of the three tokenizers).

Every distinct character in the training text becomes one token.
Tiny vocabulary, no real unknown word problem, but sequences are long
because one token is only one character.
"""

from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def train(self, text):
        self._init_specials()
        # sorted so the vocabulary (and ids) are identical on every run
        for ch in sorted(set(text)):
            self._add_token(ch)

    def _encode(self, text):
        unk = self.unk_id
        return [self.token_to_id.get(ch, unk) for ch in text]

    def _detokenize(self, tokens):
        return "".join(tokens)
