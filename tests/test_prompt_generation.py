"""python -m pytest tests/test_prompt_generation.py -v"""
import pytest
import torch

from src.generator import (GenConfig, generate_continuation, generate_ids, generate_story,
                           prepare_prompt_ids)
from src.model import ModelConfig, TransformerLM
from tokenizers import BPETokenizer, CharTokenizer, WordTokenizer

TEXT = "once upon a time there was a little fox who lived near the river. " * 5


class PTok:
    bos_id, eos_id, unk_id, vocab_size = 2, 3, 1, 50

    def encode(self, text, add_special_tokens=False):
        return [4 + (ord(c) % 40) for c in text]

    def decode(self, ids):
        return " ".join(map(str, ids))


def model(block=8):
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=50, block_size=block, n_embd=16,
                                     n_head=2, n_layer=1)).eval()


def force(m, token):
    m.lm_head.weight.data.zero_()
    m.lm_head.bias.data.zero_()
    m.lm_head.bias.data[token] = 100.0


def spy(m):
    seen = []
    orig = m.forward
    m.forward = lambda x, *a, **k: (seen.append((x[0].tolist(), torch.is_grad_enabled(), m.training)),
                                    orig(x, *a, **k))[1]
    return seen


def test_unconditional_unchanged():
    m = model()
    seen = spy(m)
    a = generate_ids(m, PTok(), GenConfig(max_new_tokens=5, seed=3))
    assert seen[0][0] == [2]                                   # starts at BOS only
    assert a == generate_ids(m, PTok(), GenConfig(max_new_tokens=5, seed=3), prompt=None)


def test_prompt_is_context_and_excluded_from_output():
    m = model(block=32)
    force(m, 10)
    seen = spy(m)
    out = generate_ids(m, PTok(), GenConfig(max_new_tokens=3, greedy=True), prompt="abc")
    assert seen[0][0] == [2] + PTok().encode("abc")
    assert out == [10, 10, 10]                                 # no prompt ids returned


def test_eos_and_max_new_tokens_count_only_new_tokens():
    m = model(block=32)
    force(m, 3)
    assert generate_ids(m, PTok(), GenConfig(greedy=True), prompt="hello there") == []
    force(m, 10)
    assert len(generate_ids(m, PTok(), GenConfig(max_new_tokens=5, greedy=True),
                            prompt="a very long opening " * 1)) == 5


def test_long_prompt_keeps_bos_and_newest_tokens():
    enc = PTok().encode("x" * 5 + "abcdefghijklmnopqrstuvwxyz")
    ids, info = prepare_prompt_ids(PTok(), "x" * 5 + "abcdefghijklmnopqrstuvwxyz", 8)
    assert ids[0] == 2 and len(ids) == 8 and ids[1:] == enc[-7:]
    assert info == {"prompt_tokens_original": len(enc), "prompt_tokens_used": 7,
                    "truncated": True, "bos_prepended": True}


def test_context_never_exceeds_block_with_long_prompt():
    m = model(block=8)
    seen = spy(m)
    generate_ids(m, PTok(), GenConfig(max_new_tokens=10, greedy=True), prompt="a" * 40)
    assert max(len(s[0]) for s in seen) <= 8


@pytest.mark.parametrize("make", [lambda: CharTokenizer(), lambda: WordTokenizer(),
                                  lambda: BPETokenizer(vocab_size=100)])
def test_real_tokenizer_conventions(make):
    tok = make()
    tok.train(TEXT)
    ids, info = prepare_prompt_ids(tok, "once upon a time", 256)
    assert ids[0] == tok.bos_id and tok.eos_id not in ids
    assert info["prompt_tokens_original"] == len(tok.encode("once upon a time"))
    assert not info["truncated"]


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_prompt_rejected(bad):
    with pytest.raises(ValueError):
        prepare_prompt_ids(PTok(), bad, 8)


def test_all_unknown_prompt_rejected():
    tok = CharTokenizer()
    tok.train(TEXT)
    with pytest.raises(ValueError, match="usable"):
        prepare_prompt_ids(tok, "\u2603\u2603", 256)


def test_malformed_tokenizer_output_rejected():
    class Bad(PTok):
        def encode(self, text, add_special_tokens=False):
            return [999]

    class Bad2(PTok):
        def encode(self, text, add_special_tokens=False):
            return ["a", "b"]

    for t in (Bad(), Bad2()):
        with pytest.raises(ValueError):
            prepare_prompt_ids(t, "hi", 8)


def test_eval_mode_no_grad_and_reproducible():
    m = model(block=32)
    m.train()
    seen = spy(m)
    a = generate_ids(m, PTok(), GenConfig(max_new_tokens=6, seed=7), prompt="hello")
    assert all(g is False and t is False for _, g, t in seen)
    assert a == generate_ids(m, PTok(), GenConfig(max_new_tokens=6, seed=7), prompt="hello")


def test_generate_story_and_continuation_fields():
    m = model(block=32)
    force(m, 10)
    gen = GenConfig(max_new_tokens=4, greedy=True)
    text, n, secs = generate_story(m, PTok(), gen, prompt="hello")
    assert text == "10 10 10 10" and n == 4 and secs >= 0
    out = generate_continuation(m, PTok(), gen, "hello")
    assert out["text"] == text and out["num_tokens"] == 4
    assert out["total_seconds"] >= out["generation_seconds"] >= 0
    assert out["prompt_tokens_used"] == 5