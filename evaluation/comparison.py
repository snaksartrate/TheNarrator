"""Six-row model comparison. Missing results stay visible as NaN/blank + status."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .schemas import EXPERIMENT_IDS, RESULTS_DIR

STORY_MEANS = ["word_count", "distinct_1", "distinct_2", "repetition_rate", "lexical_diversity",
               "generation_time_seconds", "seconds_per_word", "words_per_second"]


def _csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()


def load_perplexity(results_dir=RESULTS_DIR):
    p = Path(results_dir) / "perplexity.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build_comparison(outputs_dir=config.OUTPUT_DIR, results_dir=RESULTS_DIR):
    story = _csv(Path(results_dir) / "automatic_evaluation.csv")
    human = _csv(Path(results_dir) / "human_summary.csv")
    ppl = load_perplexity(results_dir)
    rows = []
    for eid in EXPERIMENT_IDS:
        opt, tok = config.EXPERIMENTS[eid]
        r = {"experiment_id": eid, "optimizer": opt, "tokenizer": tok}
        mp = Path(outputs_dir) / eid / "run_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            r.update(training_status=m.get("status"), steps_completed=m.get("steps_completed"),
                     vocab_size=m.get("tokenizer", {}).get("vocab_size"),
                     best_val_loss=m.get("best_val_loss"), best_step=m.get("best_step"),
                     train_seconds=m.get("train_seconds"))
        else:
            r["training_status"] = "pending"
        sub = story[story["experiment_id"] == eid] if len(story) else story
        r["n_stories"] = len(sub)
        for c in STORY_MEANS:
            r[f"mean_{c}"] = pd.to_numeric(sub[c], errors="coerce").mean() if len(sub) else np.nan
        r["median_word_count"] = sub["word_count"].median() if len(sub) else np.nan
        p = ppl.get(eid, {})
        for k in ("test_mean_cross_entropy", "test_perplexity", "test_bits_per_token",
                  "test_unknown_token_rate"):
            r[k] = p.get(k, np.nan)
        h = human[human["experiment_id"] == eid] if len(human) else human
        hr = h.iloc[0].to_dict() if len(h) else {}
        for k in ("n_ratings", "n_evaluators", "average_score_mean"):
            r[f"human_{k}"] = hr.get(k, np.nan)
        for c in ("grammar", "coherence", "creativity", "continuity", "relevance", "completeness"):
            r[f"human_{c}_mean"] = hr.get(f"{c}_mean", np.nan)
        missing = [n for n, ok in (("stories", len(sub) > 0), ("perplexity", bool(p)),
                                   ("human", bool(hr) and hr.get("n_ratings", 0) > 0)) if not ok]
        r["missing"] = ";".join(missing)
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(results_dir) / "model_comparison.csv", index=False)
    return df