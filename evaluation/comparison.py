"""Six-row model comparison. Missing results stay visible as NaN/blank + status."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .schemas import EXPERIMENT_IDS, RESULTS_DIR

STORY_MEANS = ["word_count", "distinct_1", "distinct_2", "repetition_rate", "lexical_diversity",
               "generation_time_seconds", "seconds_per_word", "words_per_second"]
CRITERIA = ("grammar", "coherence", "creativity", "continuity", "relevance", "completeness")


def _csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()


def load_perplexity(results_dir=RESULTS_DIR):
    p = Path(results_dir) / "perplexity.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _num(sub, c):
    return pd.to_numeric(sub[c], errors="coerce") if len(sub) else pd.Series(dtype=float)


def build_comparison(outputs_dir=config.OUTPUT_DIR, results_dir=RESULTS_DIR):
    story = _csv(Path(results_dir) / "automatic_evaluation.csv")
    human = _csv(Path(results_dir) / "human_summary.csv")
    ppl = load_perplexity(results_dir)
    rows = []
    for eid in EXPERIMENT_IDS:
        opt, tok = config.EXPERIMENTS[eid]
        r = {"experiment_id": eid, "optimizer": opt, "tokenizer": tok,
             "training_status": "pending", "steps_completed": np.nan, "vocab_size": np.nan,
             "best_val_loss": np.nan, "best_step": np.nan, "train_seconds": np.nan}
        mp = Path(outputs_dir) / eid / "run_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            r.update(training_status=m.get("status"), steps_completed=m.get("steps_completed"),
                     vocab_size=m.get("tokenizer", {}).get("vocab_size"),
                     best_val_loss=m.get("best_val_loss"), best_step=m.get("best_step"),
                     train_seconds=m.get("train_seconds"))
        sub = story[story["experiment_id"] == eid] if len(story) else story
        r["n_stories"] = len(sub)
        for c in STORY_MEANS:
            r[f"mean_{c}"] = _num(sub, c).mean() if len(sub) else np.nan
        wc = _num(sub, "word_count")
        gt = _num(sub, "generation_time_seconds")
        r["median_word_count"] = wc.median() if len(wc) else np.nan
        r["median_generation_time_seconds"] = gt.median() if len(gt) else np.nan
        for name, col in (("word_count", wc), ("generation_time_seconds", gt)):
            r[f"std_{name}"] = col.std(ddof=1) if len(col) > 1 else np.nan
            r[f"min_{name}"] = col.min() if len(col) else np.nan
            r[f"max_{name}"] = col.max() if len(col) else np.nan
        p = ppl.get(eid, {})
        for k in ("test_mean_cross_entropy", "test_perplexity", "test_bits_per_token",
                  "test_unknown_token_rate"):
            r[k] = p.get(k, np.nan)
        h = human[human["experiment_id"] == eid] if len(human) else human
        hr = h.iloc[0].to_dict() if len(h) else {}
        for k in ("n_ratings", "n_stories", "n_evaluators", "average_score_mean",
                  "average_score_std"):
            r[f"human_{k}"] = hr.get(k, np.nan)
        for c in CRITERIA:
            r[f"human_{c}_mean"] = hr.get(f"{c}_mean", np.nan)
        missing = [n for n, ok in (("stories", len(sub) > 0), ("perplexity", bool(p)),
                                   ("human", bool(hr) and hr.get("n_ratings", 0) > 0)) if not ok]
        r["missing"] = ";".join(missing)
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(results_dir) / "model_comparison.csv", index=False)
    return df