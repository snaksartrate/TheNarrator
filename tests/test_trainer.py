"""Run from the repository root: python -m pytest tests/test_trainer.py -v"""
import csv
import math
import random

import pytest
import torch

from src.data import STORY_SEP, StoryDataset
from src.model import ModelConfig, TransformerLM
from src.trainer import (TrainConfig, WindowSampler, build_model_from_checkpoint,
                         build_split_dataset, build_token_stream, create_optimizer,
                         evaluate, load_checkpoint, set_seed, train)


class FakeTokenizer:
    """Stand-in with the real tokenizer interface (BOS=2, EOS=3)."""
    vocab_size = 50

    def encode(self, text, add_special_tokens=False):
        ids = [4 + (ord(c) % 46) for c in text]
        return [2] + ids + [3] if add_special_tokens else ids


def tiny_model():
    return TransformerLM(ModelConfig(vocab_size=50, block_size=8, n_embd=16,
                                     n_head=2, n_layer=1, dropout=0.1))


def tiny_data():
    r = random.Random(1)
    tr = StoryDataset([r.randrange(4, 50) for _ in range(800)], 8)   # 99 windows
    va = StoryDataset([r.randrange(4, 50) for _ in range(300)], 8)   # 37 windows
    return tr, va


TOK = {"name": "fake", "path": "none", "vocab_size": 50}


def run(out_dir, opt="adamw", max_steps=6, lr=None, exp="adamw_tiny", model_seed=0, **kw):
    set_seed(model_seed)
    model = tiny_model()
    tr, va = tiny_data()
    cfg = TrainConfig(experiment_id=exp, optimizer=opt,
                      lr=lr if lr is not None else (3e-4 if opt == "adamw" else 0.1),
                      batch_size=4, max_steps=max_steps, eval_every=3, log_every=1,
                      eval_batches=3, seed=0, out_dir=str(out_dir))
    meta = train(model, tr, va, cfg, TOK, log=lambda *_: None, **kw)
    return model, meta


# ---------------- data ----------------
def test_stream_has_bos_eos_per_story():
    assert build_token_stream(["ab", "cde"], FakeTokenizer()) == [
        2, *FakeTokenizer().encode("ab"), 3, 2, *FakeTokenizer().encode("cde"), 3]


def test_split_file_delimiter_not_encoded(tmp_path):
    (tmp_path / "train.txt").write_text(
        f"Story one.\n{STORY_SEP}\nStory two.\n{STORY_SEP}\n", encoding="utf-8")
    ds = build_split_dataset("train", FakeTokenizer(), 4, processed_dir=tmp_path)
    assert len(ds.ids) == (len("Story one.") + 2) * 2        # no separator text encoded
    assert (ds.ids == 2).sum() == 2 and (ds.ids == 3).sum() == 2


def test_shift_shapes_and_dtype():
    ds = StoryDataset(list(range(20)), 4)
    x, y = ds[1]
    assert x.tolist() == [4, 5, 6, 7] and y.tolist() == [5, 6, 7, 8]
    assert x.dtype == torch.long and y.dtype == torch.long
    assert all(len(ds[i][0]) == 4 == len(ds[i][1]) for i in range(len(ds)))


# ---------------- sampler / optimizer / config ----------------
def test_sampler_reproducible_and_without_replacement():
    a, b = WindowSampler(100, 8, 42), WindowSampler(100, 8, 42)
    seen = [i for s in range(12) for i in a.batch_indices(s)]
    assert len(seen) == len(set(seen)) == 96
    assert b.batch_indices(7) == a.batch_indices(7)           # random access, same batch
    assert a.batch_indices(12) != a.batch_indices(0)          # new pass, new shuffle


def test_optimizer_factory():
    m = tiny_model()
    a = create_optimizer(m, TrainConfig("x", "adamw", lr=3e-4))
    s = create_optimizer(m, TrainConfig("x", "sgd", lr=0.1, momentum=0.9))
    assert isinstance(a, torch.optim.AdamW) and a.param_groups[0]["lr"] == 3e-4
    assert isinstance(s, torch.optim.SGD)
    assert s.param_groups[0]["momentum"] == 0.9 and s.param_groups[0]["lr"] == 0.1
    with pytest.raises(ValueError):
        TrainConfig("x", "rmsprop", lr=0.1)


def test_evaluate_no_grad_and_restores_mode():
    model = tiny_model()
    _, va = tiny_data()
    seen = []
    orig = model.forward

    def spy(*a, **k):
        seen.append((torch.is_grad_enabled(), model.training))
        return orig(*a, **k)

    model.forward = spy
    model.train()
    loss = evaluate(model, va, list(range(6)), 2, torch.device("cpu"))
    assert math.isfinite(loss) and model.training
    assert seen and all(g is False and t is False for g, t in seen)
    assert all(p.grad is None for p in model.parameters())


# ---------------- training run ----------------
@pytest.mark.parametrize("opt", ["adamw", "sgd"])
def test_tiny_run_files_and_history(tmp_path, opt):
    _, meta = run(tmp_path, opt=opt, exp=f"{opt}_tiny")
    d = tmp_path / f"{opt}_tiny"
    assert all((d / f).exists() for f in ("last.pt", "best.pt", "history.csv", "run_meta.json"))
    assert meta["status"] == "completed" and meta["steps_completed"] == 6
    assert meta["windows_drawn"] == 24 and meta["seconds_per_100_steps"] > 0
    rows = list(csv.DictReader(open(d / "history.csv", encoding="utf-8")))
    assert len(rows) == 6
    assert [r["step"] for r in rows if r["val_loss"]] == ["3", "6"]
    assert all(math.isfinite(float(r["train_loss"])) for r in rows)


def test_checkpoint_reload_same_outputs_and_strict(tmp_path):
    model, _ = run(tmp_path)
    ckpt = load_checkpoint(tmp_path / "adamw_tiny" / "last.pt")
    model2 = build_model_from_checkpoint(ckpt)
    x = torch.randint(0, 50, (2, 8))
    model.eval(), model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)[0], model2(x)[0], atol=1e-6)
    del ckpt["model_state_dict"]["lm_head.bias"]
    with pytest.raises(RuntimeError):
        build_model_from_checkpoint(ckpt)


def test_resume_matches_uninterrupted(tmp_path):
    full, _ = run(tmp_path / "a", max_steps=6)
    run(tmp_path / "b", max_steps=3)                              # "interrupted" at step 3
    resumed, meta = run(tmp_path / "b", max_steps=6, resume=True, model_seed=123)
    assert meta["steps_completed"] == 6
    for (k, p), (_, q) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(p, q, atol=1e-5), k


def test_resume_mismatch_and_existing_run(tmp_path):
    run(tmp_path, max_steps=3)
    with pytest.raises(ValueError, match="optimizer_settings"):
        run(tmp_path, max_steps=6, lr=0.5, resume=True)
    with pytest.raises(FileExistsError):
        run(tmp_path, max_steps=3)


def test_divergence_is_recorded_not_raised(tmp_path):
    _, meta = run(tmp_path, opt="sgd", lr=float("inf"), exp="sgd_div")
    assert meta["status"] == "diverged" and meta["steps_completed"] < 6
    assert (tmp_path / "sgd_div" / "history.csv").exists()
    assert (tmp_path / "sgd_div" / "run_meta.json").exists()