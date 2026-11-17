"""python -m pytest tests/test_evaluation.py -v   (synthetic fixtures only)"""
import json
import math

import pytest
import torch
import torch.nn.functional as F

from evaluation import human, metrics, perplexity, runner
from evaluation.comparison import build_comparison
from evaluation.schemas import CRITERIA, EXPERIMENT_IDS
from src.model import ModelConfig, TransformerLM


# ---------- metrics (hand-computed) ----------
def test_empty_and_punct():
    for t in ("", "   ", "...!?"):
        m = metrics.story_metrics(t)
        assert m["word_count"] == 0 and m["distinct_1"] == 0 and m["repetition_rate"] == 0


def test_tokenization_rules():
    assert metrics.tokenize_words("Don't stop, ok-ay! 42") == ["don't", "stop", "ok", "ay", "42"]
    assert metrics.tokenize_words("CAT cat") == ["cat", "cat"]


def test_values():
    assert metrics.distinct_n("cat dog cat", 1) == pytest.approx(2 / 3)
    assert metrics.distinct_n("cat dog cat", 2) == 1.0
    assert metrics.distinct_n("hi", 2) == 0.0
    assert metrics.repetition_rate("a b c a b c") == pytest.approx(0.25)
    assert metrics.repetition_rate("a a a a a") == pytest.approx(2 / 3)   # overlapping trigrams
    assert metrics.lexical_diversity("cat dog cat") == pytest.approx(2 / 3)


# ---------- perplexity ----------
def tiny():
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=20, block_size=16, n_embd=16, n_head=2, n_layer=1))


def test_windows_cover_every_target_once():
    ids = list(range(2, 13))                      # 11 tokens -> 10 targets
    w = list(perplexity.build_windows([ids], 4))
    assert sum(len(y) for _, y in w) == 10
    assert [t for _, y in w for t in y] == ids[1:]
    assert all(x == ids[s:s + len(x)] for (x, _), s in zip(w, range(0, 10, 4)))


def test_matches_manual_full_context_and_is_batch_invariant():
    m = tiny().eval()
    stories = [[2, 5, 6, 7, 3], [2, 9, 8, 3], [2, 4, 5, 6, 7, 8, 9, 3]]
    with torch.no_grad():
        manual = 0.0
        for s in stories:
            x, y = torch.tensor([s[:-1]]), torch.tensor([s[1:]])
            manual += F.cross_entropy(m(x)[0].reshape(-1, 20), y.reshape(-1), reduction="sum").item()
    r1 = perplexity.sequence_nll(m, stories, batch_size=1)
    r2 = perplexity.sequence_nll(m, stories, batch_size=2)     # uneven last batch
    assert r1["token_count"] == 4 + 3 + 7
    assert r1["total_nll"] == pytest.approx(manual, rel=1e-5)
    assert r2["total_nll"] == pytest.approx(r1["total_nll"], rel=1e-5)
    s = perplexity.summarize(r1)
    assert s["test_perplexity"] == pytest.approx(math.exp(manual / 14), rel=1e-5)
    assert s["test_bits_per_token"] == pytest.approx(s["test_mean_cross_entropy"] / math.log(2))


def test_eval_mode_restored_no_grad_deterministic_unk():
    m = tiny().train()
    a = perplexity.sequence_nll(m, [[2, 1, 1, 3]], unk_id=1)
    b = perplexity.sequence_nll(m, [[2, 1, 1, 3]], unk_id=1)
    assert m.training and a == b and a["unk_count"] == 2
    assert all(p.grad is None for p in m.parameters())


def test_nonfinite_raises():
    m = tiny().eval()
    m.lm_head.bias.data[:] = float("nan")
    with pytest.raises(FloatingPointError):
        perplexity.sequence_nll(m, [[2, 5, 3]])


# ---------- runner ----------
def make_run(root, eid="adamw_char", tok="char", opt="AdamW", smoke=False, texts=("a b c a b c", "x y z")):
    d = root / eid / "generations"
    d.mkdir(parents=True)
    (root / eid / "run_meta.json").write_text(json.dumps(
        {"smoke_test": smoke, "tokenizer": {"vocab_size": 84}, "device": "cpu", "dataset": "d"}))
    stories = []
    for i, t in enumerate(texts):
        (d / f"story_{i:02d}.txt").write_text(t)
        stories.append({"file": f"story_{i:02d}.txt", "seed": 42 + i, "num_tokens": 10, "seconds": 2.0})
    (d / "generation_meta.json").write_text(json.dumps(
        {"experiment_id": eid, "tokenizer": tok, "optimizer": opt, "checkpoint": "best.pt",
         "checkpoint_step": 5, "generation_settings": {"seed": 42}, "stories": stories}))


def test_runner_rows_idempotent_and_exclusions(tmp_path):
    out, res = tmp_path / "o", tmp_path / "r"
    make_run(out)
    make_run(out, "sgd_bpe", "bpe", "SGD", smoke=True)
    rows, problems = runner.evaluate_stories(out, res)
    assert len(rows) == 2 and "sgd_bpe" in problems and "adamw_word" in problems
    assert rows[0]["seconds_per_word"] == pytest.approx(2.0 / 6)
    first = (res / "automatic_evaluation.csv").read_text()
    runner.evaluate_stories(out, res)
    assert (res / "automatic_evaluation.csv").read_text() == first      # no duplicates
    assert json.loads((res / "automatic_evaluation.json").read_text()) == json.loads(
        json.dumps(rows))


def test_runner_mismatch_and_missing_story(tmp_path):
    make_run(tmp_path, "adamw_char", tok="word")
    with pytest.raises(runner.ArtifactError):
        runner.collect_run(tmp_path / "adamw_char", "adamw_char")
    (tmp_path / "sgd_char").mkdir()
    with pytest.raises(runner.ArtifactError):
        runner.collect_run(tmp_path / "sgd_char", "sgd_char")
    with pytest.raises(runner.ArtifactError):
        runner.collect_run(tmp_path, "dummy_gru")


def test_zero_word_story_no_division_error(tmp_path):
    make_run(tmp_path, texts=("!!!",))
    rows = runner.collect_run(tmp_path / "adamw_char", "adamw_char")
    assert rows[0]["seconds_per_word"] is None


# ---------- human ----------
def good(**kw):
    r = {"evaluator_id": "e1", "blind_story_id": "B001", **{c: 3 for c in CRITERIA}}
    r.update(kw)
    return r


@pytest.mark.parametrize("bad", [{"grammar": 0}, {"grammar": 6}, {"grammar": float("nan")},
                                 {"grammar": "x"}, {"evaluator_id": ""}])
def test_human_validation_rejects(bad):
    with pytest.raises(ValueError):
        human.validate_rating(good(**bad))


def test_human_pipeline(tmp_path):
    out = tmp_path / "o"
    make_run(out)
    assert human.make_blind_sheet(out, tmp_path / "r") == 2
    key = tmp_path / "r" / "blind_key.csv"
    import pandas as pd
    ids = pd.read_csv(key)["blind_story_id"].tolist()
    pd.DataFrame([good(blind_story_id=ids[0], grammar=5), good(blind_story_id=ids[1]),
                  good(evaluator_id="e2", blind_story_id=ids[0], grammar=3)]
                 ).to_csv(tmp_path / "rat.csv", index=False)
    s = human.run_human_summary(tmp_path / "rat.csv", key, tmp_path / "r")
    assert len(s) == 6 and s.loc[s.experiment_id == "adamw_char", "n_evaluators"].iloc[0] == 2
    assert s.loc[s.experiment_id == "sgd_bpe", "n_ratings"].iloc[0] == 0
    pd.DataFrame([good(blind_story_id=ids[0])] * 2).to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError):
        human.load_ratings(tmp_path / "dup.csv", key)


# ---------- comparison ----------
def test_comparison_always_six_rows(tmp_path):
    make_run(tmp_path / "o")
    runner.evaluate_stories(tmp_path / "o", tmp_path / "r")
    df = build_comparison(tmp_path / "o", tmp_path / "r")
    assert df["experiment_id"].tolist() == EXPERIMENT_IDS
    assert df.loc[0, "n_stories"] == 2 and df.loc[5, "training_status"] == "pending"
    assert math.isnan(df.loc[5, "mean_word_count"])           # missing is NaN, not 0