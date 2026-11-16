"""Run from the repository root: python -m pytest tests/test_generator.py -v"""
import json

import pytest
import torch

from src.generator import GenConfig, generate_and_save, generate_ids
from src.model import ModelConfig, TransformerLM
from src.trainer import save_checkpoint
from tokenizers import CharTokenizer


class FakeTok:
    bos_id, eos_id = 2, 3

    def decode(self, ids):
        return " ".join(map(str, ids))


def model(block=8):
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=50, block_size=block, n_embd=16,
                                     n_head=2, n_layer=1)).eval()


def force(m, token):
    """Make the model always predict `token`."""
    m.lm_head.weight.data.zero_()
    m.lm_head.bias.data.zero_()
    m.lm_head.bias.data[token] = 100.0


def test_stops_at_eos():
    m = model()
    force(m, 3)
    assert generate_ids(m, FakeTok(), GenConfig(greedy=True)) == []


def test_stops_at_max_new_tokens():
    m = model()
    force(m, 10)
    assert generate_ids(m, FakeTok(), GenConfig(max_new_tokens=5, greedy=True)) == [10] * 5


def test_context_never_exceeds_block_size():
    m = model(block=8)
    lengths, orig = [], m.forward
    m.forward = lambda x, *a, **k: (lengths.append(x.size(1)), orig(x, *a, **k))[1]
    generate_ids(m, FakeTok(), GenConfig(max_new_tokens=30, greedy=True))
    assert lengths and max(lengths) <= 8


def test_seed_reproducible_and_topk1_is_greedy():
    m = model()
    a = generate_ids(m, FakeTok(), GenConfig(max_new_tokens=20, seed=1))
    assert a == generate_ids(m, FakeTok(), GenConfig(max_new_tokens=20, seed=1))
    assert a != generate_ids(m, FakeTok(), GenConfig(max_new_tokens=20, seed=2))
    g = generate_ids(m, FakeTok(), GenConfig(max_new_tokens=20, greedy=True))
    assert g == generate_ids(m, FakeTok(), GenConfig(max_new_tokens=20, top_k=1))


@pytest.mark.parametrize("kw", [{"temperature": 0}, {"top_k": 0}, {"top_k": 51}])
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        generate_ids(model(), FakeTok(), GenConfig(**kw))


def test_load_run_and_save_outputs(tmp_path):
    tok = CharTokenizer()
    tok.train("hello world, once upon a time.")
    tok.save(tmp_path / "tok.json")
    m = TransformerLM(ModelConfig(vocab_size=tok.vocab_size, block_size=8,
                                  n_embd=16, n_head=2, n_layer=1))
    run = tmp_path / "adamw_char"
    run.mkdir()
    save_checkpoint(run / "best.pt", {
        "model_state_dict": m.state_dict(), "model_config": m.cfg.to_dict(),
        "tokenizer": {"name": "char", "path": str(tmp_path / "tok.json"),
                      "vocab_size": tok.vocab_size},
        "experiment_id": "adamw_char", "optimizer_settings": {"name": "AdamW"}, "step": 7})
    meta = generate_and_save(run, GenConfig(max_new_tokens=10), n_stories=2)
    assert meta["experiment_id"] == "adamw_char" and len(meta["stories"]) == 2
    assert (run / "generations" / "story_01.txt").exists()
    saved = json.loads((run / "generations" / "generation_meta.json").read_text())
    assert saved["optimizer"] == "AdamW" and saved["tokenizer"] == "char"