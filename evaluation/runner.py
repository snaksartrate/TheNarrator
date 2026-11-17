"""Evaluate SAVED generated stories. Rebuilds the table from artifacts every run,
so re-running is deterministic and can never append duplicate rows."""
import csv
import json
import os
from pathlib import Path

import config
from .metrics import story_metrics
from .schemas import EXPERIMENT_IDS, RESULTS_DIR, STORY_COLUMNS


class ArtifactError(ValueError):
    pass


def _read_json(path):
    if not Path(path).exists():
        raise ArtifactError(f"missing {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_run(run_dir, experiment_id):
    """Return one record per generated story of one run, or raise ArtifactError."""
    if experiment_id not in config.EXPERIMENTS:
        raise ArtifactError(f"unknown experiment id {experiment_id!r}")
    opt, tok = config.EXPERIMENTS[experiment_id]
    run_dir = Path(run_dir)
    run_meta = _read_json(run_dir / "run_meta.json")
    if run_meta.get("smoke_test"):
        raise ArtifactError(f"{experiment_id}: smoke-test run, excluded from results")
    gen = _read_json(run_dir / "generations" / "generation_meta.json")
    if gen.get("experiment_id") != experiment_id:
        raise ArtifactError(f"{experiment_id}: generation_meta experiment_id={gen.get('experiment_id')!r}")
    if gen.get("tokenizer") != tok:
        raise ArtifactError(f"{experiment_id}: tokenizer {gen.get('tokenizer')!r} != {tok!r}")
    if str(gen.get("optimizer", "")).lower() != opt:
        raise ArtifactError(f"{experiment_id}: optimizer {gen.get('optimizer')!r} != {opt!r}")
    if not gen.get("stories"):
        raise ArtifactError(f"{experiment_id}: no stories listed")

    rows = []
    for s in gen["stories"]:
        f = run_dir / "generations" / s["file"]
        if not f.exists():
            raise ArtifactError(f"{experiment_id}: story file missing: {f}")
        text = f.read_text(encoding="utf-8")
        # a blank story is a real (degenerate) model output: keep it, metrics will be 0 words
        m = story_metrics(text)
        secs = s.get("seconds")
        wc = m["word_count"]
        rows.append({
            "experiment_id": experiment_id, "optimizer": opt, "tokenizer": tok,
            "vocab_size": run_meta.get("tokenizer", {}).get("vocab_size"),
            "checkpoint": gen.get("checkpoint"), "checkpoint_step": gen.get("checkpoint_step"),
            "story_id": Path(s["file"]).stem, "story_file": s["file"],
            "generation_seed": s.get("seed"),
            "generation_settings": json.dumps(gen.get("generation_settings"), sort_keys=True),
            "dataset": run_meta.get("dataset"), "device": run_meta.get("device"),
            **m,
            "generation_time_seconds": secs,
            "seconds_per_word": (secs / wc) if (secs is not None and wc > 0) else None,
            "words_per_second": (wc / secs) if (secs and wc > 0) else None,
            "generated_token_count": s.get("num_tokens"),
        })
    return rows


def _atomic_write(path, write_fn):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


def write_story_results(rows, results_dir=RESULTS_DIR):
    order = {e: i for i, e in enumerate(EXPERIMENT_IDS)}
    rows = sorted(rows, key=lambda r: (order[r["experiment_id"]], r["story_id"]))
    keys = [(r["experiment_id"], r["story_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ArtifactError("duplicate (experiment_id, story_id) rows")

    def w_csv(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STORY_COLUMNS, extrasaction="raise")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r[k] is None else r[k]) for k in STORY_COLUMNS})

    def w_json(p):
        Path(p).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    _atomic_write(Path(results_dir) / "automatic_evaluation.csv", w_csv)
    _atomic_write(Path(results_dir) / "automatic_evaluation.json", w_json)
    return rows


def evaluate_stories(outputs_dir=config.OUTPUT_DIR, results_dir=RESULTS_DIR, experiments=None):
    """Returns (rows, problems). Failed/pending runs go in problems, never in rows."""
    rows, problems = [], {}
    for eid in experiments or EXPERIMENT_IDS:
        run_dir = Path(outputs_dir) / eid
        if not run_dir.exists():
            problems[eid] = "pending: run folder not found"
            continue
        try:
            rows.extend(collect_run(run_dir, eid))
        except ArtifactError as e:
            problems[eid] = str(e)
    write_story_results(rows, results_dir)
    return rows, problems