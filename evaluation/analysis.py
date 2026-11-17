"""Numbers-only tables answering the plan's analysis questions. No narrative:
conclusions are written by a human from these tables."""
from pathlib import Path

import pandas as pd

from .schemas import RESULTS_DIR

METRICS = ["best_val_loss", "test_perplexity", "mean_word_count", "mean_distinct_1",
           "mean_distinct_2", "mean_repetition_rate", "mean_generation_time_seconds",
           "mean_words_per_second", "human_average_score_mean"]
HIGHER_BETTER = {"mean_distinct_1": True, "mean_distinct_2": True, "mean_repetition_rate": False,
                 "human_average_score_mean": True}
CAVEAT = ("Loss/perplexity are token-space: compare only within a tokenizer. Diversity and "
          "repetition depend on story length (see word count). 5 stories/model: descriptive only.")


def _block(title, df):
    return [f"## {title}", "```", df.to_string(float_format=lambda v: f"{v:.4f}"), "```", ""]


def build_analysis(results_dir=RESULTS_DIR):
    d = Path(results_dir)
    comp = pd.read_csv(d / "model_comparison.csv")
    cols = [m for m in METRICS if m in comp]
    out = ["# Analysis tables (generated, numbers only)", "", f"> {CAVEAT}", ""]
    for tok in ("char", "word", "bpe"):                      # Q1-3
        sub = comp[comp.tokenizer == tok].set_index("optimizer")[cols].T
        if {"adamw", "sgd"} <= set(sub.columns):
            sub["sgd_minus_adamw"] = sub["sgd"] - sub["adamw"]
        out += _block(f"Within {tok}: AdamW vs SGD", sub)
    for opt in ("adamw", "sgd"):                             # Q4-5
        sub = comp[comp.optimizer == opt].set_index("tokenizer")[cols].T
        out += _block(f"Within {opt}: across tokenizers", sub)
    ranks = []                                               # Q6-7, Q12
    for m, hi in HIGHER_BETTER.items():
        s = comp.set_index("experiment_id")[m].dropna() if m in comp else pd.Series(dtype=float)
        if len(s):
            ranks.append(f"{m}: best={s.idxmax() if hi else s.idxmin()} "
                         f"({'highest' if hi else 'lowest'}), n_models={len(s)}")
    out += ["## Best model per metric", "```", *ranks, "```", ""]
    sp = d / "automatic_evaluation.csv"                      # Q8
    if sp.exists() and sp.stat().st_size:
        st = pd.read_csv(sp)
        if len(st) > 2:
            rows = {c: st["word_count"].corr(st[c], method="spearman")
                    for c in ("distinct_1", "distinct_2", "repetition_rate")}
            out += _block("Spearman(word_count, metric), pooled stories (tokenizer-confounded)",
                          pd.Series(rows, name="rho").to_frame())
    text = "\n".join(out)
    (d / "analysis_tables.md").write_text(text, encoding="utf-8")
    return text