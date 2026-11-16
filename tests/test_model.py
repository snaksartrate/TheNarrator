"""Run from the repository root: python -m pytest tests/test_model.py -v"""
import pytest
import torch

from src.model import ModelConfig, TransformerLM


def tiny(**kw):
    base = dict(vocab_size=50, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.1)
    base.update(kw)
    return ModelConfig(**base)


def batch(cfg, B=3, T=10):
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    return x, y


def test_forward_shape():
    cfg = tiny()
    x, _ = batch(cfg)
    logits, loss = TransformerLM(cfg)(x)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss is None


def test_loss_is_finite_scalar():
    cfg = tiny()
    x, y = batch(cfg)
    _, loss = TransformerLM(cfg)(x, y)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_backward_gives_gradients():
    cfg = tiny()
    x, y = batch(cfg)
    model = TransformerLM(cfg)
    _, loss = model(x, y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_causal_mask_no_future_leak():
    cfg = tiny()
    torch.manual_seed(1)
    model = TransformerLM(cfg).eval()      # eval: dropout off, deterministic
    x1 = torch.randint(0, cfg.vocab_size, (2, 12))
    x2 = x1.clone()
    x2[:, 6:] = (x2[:, 6:] + 1) % cfg.vocab_size   # change only later tokens
    with torch.no_grad():
        a, _ = model(x1)
        b, _ = model(x2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)   # earlier positions unchanged
    assert not torch.allclose(a[:, 6:], b[:, 6:])          # later positions do change


def test_bad_head_split_fails():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=50, n_embd=30, n_head=4)


def test_sequence_too_long_fails():
    cfg = tiny()
    x = torch.zeros(1, cfg.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="block_size"):
        TransformerLM(cfg)(x)


def test_config_round_trip():
    cfg = tiny()
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg