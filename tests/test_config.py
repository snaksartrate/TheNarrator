"""Run from the repository root: python -m pytest tests/test_config.py -v"""
import pytest

import config


def test_six_ids_map_to_matrix():
    assert config.EXPERIMENTS == {f"{o}_{t}": (o, t) for o in ("adamw", "sgd")
                                  for t in ("char", "word", "bpe")}


def test_approved_settings():
    assert config.OPTIMIZERS == {"adamw": {"lr": 3e-4}, "sgd": {"lr": 0.1, "momentum": 0.9}}
    assert config.SHARED_MODEL["block_size"] == 256
    t = config.TRAIN
    assert (t["batch_size"], t["max_steps"], t["eval_every"], t["grad_clip"]) == (8, 500, 100, 1.0)
    assert t["device"] == "cpu"


def test_smoke_is_separate_and_tiny():
    s = config.get_settings(smoke=True)
    assert s["out_dir"] != config.OUTPUT_DIR and s["smoke"]
    assert s["train"]["max_steps"] < config.TRAIN["max_steps"]


def test_override_and_unknown_key():
    assert config.get_settings(max_steps=2000)["train"]["max_steps"] == 2000
    with pytest.raises(ValueError):
        config.get_settings(lr=1.0)