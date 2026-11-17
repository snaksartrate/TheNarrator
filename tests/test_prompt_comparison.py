"""python -m pytest tests/test_prompt_comparison.py -v"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import config
from src import prompt_comparison as pc
from src.generator import GenConfig
from src.model import ModelConfig, TransformerLM
from src.prompt_display import render


# ---------- allocator ----------
def test_allocator_sequence_gaps_and_ignored(tmp_path):
    assert pc.allocate_run(tmp_path)[0] == 1
    (tmp_path / "run_00004").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "run_00007").write_text("a file, not a dir")
    (tmp_path / "run_7").mkdir()
    (tmp_path / "run_00000").mkdir()
    rid, d = pc.allocate_run(tmp_path)
    assert rid == 5 and d.name == "run_00005"                   # max+1, gaps not filled


def test_allocator_never_overwrites(tmp_path):
    (tmp_path / "run_00001").mkdir()
    (tmp_path / "run_00001" / "keep.txt").write_text("x")
    assert pc.allocate_run(tmp_path)[0] == 2
    assert (tmp_path / "run_00001" / "keep.txt").read_text() == "x"


def test_allocator_limit(tmp_path):
    (tmp_path / "run_99999").mkdir()
    with pytest.raises(pc.RunLimitError):
        pc.allocate_run(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run_99999"]
    assert pc.MAX_RUN_ID == 99999


def test_allocator_concurrent_unique(tmp_path):
    with ThreadPoolExecutor(8) as ex:
        ids = list(ex.map(lambda _: pc.allocate_run(tmp_path)[0], range(16)))
    assert sorted(ids) == list(range(1, 17))


# ---------- orchestration with fake models ----------
class FTok:
    bos_id, eos_id, unk_id, vocab_size = 2, 3, 1, 50

    def encode(self, text, add_special_tokens=False):
        return [4 + (ord(c) % 40) for c in text]

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def fake_loader(fail=()):
    def load(run, checkpoint, device):
        eid = run.name
        if eid in fail:
            raise RuntimeError("corrupt checkpoint")
        torch.manual_seed(0)
        m = TransformerLM(ModelConfig(vocab_size=50, block_size=32, n_embd=16, n_head=2, n_layer=1)).eval()
        m.lm_head.weight.data.zero_()
        m.lm_head.bias.data.zero_()
        m.lm_head.bias.data[10] = 100.0
        return m, FTok(), {"experiment_id": eid, "tokenizer": {"name": config.EXPERIMENTS[eid][1]}, "step": 7}
    return load


def make_outputs(tmp_path, missing=()):
    out = tmp_path / "outputs"
    for eid in config.EXPERIMENTS:
        if eid not in missing:
            (out / eid).mkdir(parents=True)
            (out / eid / "best.pt").write_bytes(b"x")
    return out


GEN = GenConfig(max_new_tokens=5, greedy=True)


def test_full_run_with_missing_and_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("src.trainer.train", lambda *a, **k: pytest.fail("training must not run"))
    out = make_outputs(tmp_path, missing=("sgd_bpe",))
    root = tmp_path / "pc"
    prompt = "A  little girl \u00e9 found a key.\n  "
    run = pc.run_comparison(prompt, GEN, outputs_dir=out, root=root,
                            loader=fake_loader(fail=("sgd_word",)))
    st = {r["experiment_id"]: r["status"] for r in run["results"]}
    assert list(st) == list(config.EXPERIMENTS)                  # approved display order
    assert st["sgd_bpe"] == "unavailable" and st["sgd_word"] == "failed"
    assert [s for s in st.values() if s == "ok"] == ["ok"] * 4
    assert run["status"] == "partial" and run["run_id"] == 1
    d = root / "run_00001"
    assert (d / "prompt_00001.txt").read_text(encoding="utf-8") == prompt   # exact prompt
    for name in ("comparison_00001.json", "comparison_00001.csv", "report_00001.md", "metadata_00001.json"):
        assert (d / name).exists()
    assert sorted(p.name for p in (d / "stories").iterdir()) == sorted(
        f"{e}_00001.txt" for e in st if st[e] == "ok")
    assert (d / "stories" / "adamw_char_00001.txt").read_text() == "w10 w10 w10 w10 w10"
    rows = (d / "comparison_00001.csv").read_text().splitlines()
    assert len(rows) == 7
    assert any(r.startswith("00001,sgd_bpe,sgd,bpe,unavailable,") for r in rows)
    assert any(r.startswith("00001,sgd_word,sgd,word,failed,RuntimeError: corrupt checkpoint") for r in rows)
    assert "corrupt checkpoint" in (d / "report_00001.md").read_text()
    # second run gets the next id and leaves the first alone
    before = (d / "report_00001.md").read_text()
    run2 = pc.run_comparison("Hello", GEN, outputs_dir=out, root=root, loader=fake_loader())
    assert run2["run_id"] == 2 and (d / "report_00001.md").read_text() == before


def test_no_checkpoints_creates_no_run(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(pc.NoCheckpointsError):
        pc.run_comparison("hi", GEN, outputs_dir=out, root=tmp_path / "pc")
    assert not (tmp_path / "pc").exists() or not list((tmp_path / "pc").iterdir())


def test_all_failed_is_marked_failed(tmp_path):
    out = make_outputs(tmp_path)
    run = pc.run_comparison("hi", GEN, outputs_dir=out, root=tmp_path / "pc",
                            loader=fake_loader(fail=tuple(config.EXPERIMENTS)))
    meta = json.loads((tmp_path / "pc" / "run_00001" / "metadata_00001.json").read_text())
    assert run["status"] == meta["status"] == "failed"


def test_single_experiment_and_no_report(tmp_path):
    out = make_outputs(tmp_path)
    run = pc.run_comparison("hi", GEN, experiments=["adamw_bpe"], outputs_dir=out,
                            root=tmp_path / "pc", loader=fake_loader(), write_report=False)
    d = tmp_path / "pc" / "run_00001"
    assert len(run["results"]) == 1 and run["meta"]["mode"] == "single/partial selection"
    assert (d / "prompt_00001.txt").exists() and not (d / "report_00001.md").exists()
    assert not (d / "comparison_00001.csv").exists()


def test_empty_prompt_rejected_at_service(tmp_path):
    with pytest.raises(ValueError):
        pc.run_comparison("   ", GEN, outputs_dir=make_outputs(tmp_path), root=tmp_path / "pc")


# ---------- display ----------
def _run(tmp_path):
    out = make_outputs(tmp_path)
    return pc.run_comparison("A little girl found a key.", GEN, outputs_dir=out,
                             root=tmp_path / "pc", loader=fake_loader())


def test_render_wide_and_narrow(tmp_path):
    run = _run(tmp_path)
    wide = render(run, width=130)
    assert "ADAMW" in wide and "SGD + MOMENTUM" in wide and "RUN 00001" in wide
    narrow = render(run, width=40)
    assert max(len(x) for x in narrow.split("\n")) <= 40
    assert "Character" in narrow and "BPE" in narrow


# ---------- CLI ----------
def _cli(monkeypatch, *args):
    import main
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    monkeypatch.setattr(main, "run_experiment", lambda *a, **k: pytest.fail("training must not run"))
    with pytest.raises(SystemExit) as e:
        main.main()
    return e.value.code


@pytest.mark.parametrize("args", [("hi", "--smoke"), ("hi", "--all"), ("hi", "--max-steps", "5"),
                                  ("hi", "--generate", "x"), ("   ",), ("--generate", ""),
                                  ("--seed", "3", "--exp", "adamw_char"), ("hi", "--temperature", "0")])
def test_cli_rejects_bad_combinations(monkeypatch, args):
    assert _cli(monkeypatch, *args) == 2


@pytest.mark.parametrize("args", [("Once upon a time",), ("--generate", "Once upon a time")])
def test_cli_dispatches_to_generation_only(monkeypatch, args, tmp_path):
    called = {}

    def fake_run(prompt, gen, **kw):
        called.update(prompt=prompt, gen=gen, kw=kw)
        return {"run_id": 1, "run_dir": tmp_path, "status": "complete", "prompt": prompt, "results": []}

    monkeypatch.setattr("src.prompt_comparison.run_comparison", fake_run)
    monkeypatch.setattr("src.prompt_display.render", lambda r: "rendered")
    assert _cli(monkeypatch, *args, "--max-new-tokens", "9", "--top-k", "5") == 0
    assert called["prompt"] == "Once upon a time"
    assert called["gen"].max_new_tokens == 9 and called["gen"].top_k == 5
    assert called["gen"].temperature == 0.8 and called["gen"].seed == 42