"""Report plots. Every function returns the saved path, or None if there is no data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schemas import CRITERIA, EXPERIMENT_IDS, VIS_DIR

_COL = {"char": "#4C72B0", "word": "#DD8452", "bpe": "#55A868"}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _color(eid):
    return _COL[eid.split("_")[1]]


def bar_by_experiment(story_df, col, ylabel, title, path):
    if story_df is None or story_df.empty or col not in story_df:
        return None
    g = story_df.groupby("experiment_id")[col].agg(["mean", "std", "count"]).reindex(EXPERIMENT_IDS)
    if g["mean"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(6), g["mean"].fillna(0), yerr=g["std"].fillna(0), capsize=4,
           color=[_color(e) for e in EXPERIMENT_IDS])
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{e}\n(n={int(n) if n == n else 0})" for e, n in zip(EXPERIMENT_IDS, g["count"])])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(mean ± SD over generated stories; small samples)")
    ax.grid(axis="y", ls="--", alpha=.4)
    return _save(fig, path)


def plot_story_metrics(story_df, vis_dir=VIS_DIR):
    specs = [("word_count", "Words", "Average story length", "word_count.png"),
             ("distinct_1", "Distinct-1", "Distinct-1", "distinct_1.png"),
             ("distinct_2", "Distinct-2", "Distinct-2", "distinct_2.png"),
             ("repetition_rate", "Repeated-trigram rate", "Repetition rate", "repetition.png"),
             ("lexical_diversity", "Unique/total words", "Lexical diversity", "lexical_diversity.png"),
             ("generation_time_seconds", "Seconds per story", "Generation time", "generation_time.png"),
             ("words_per_second", "Words / second", "Generation speed", "words_per_second.png")]
    return [bar_by_experiment(story_df, c, y, t, Path(vis_dir) / f) for c, y, t, f in specs]


def plot_training_curves(outputs_dir, vis_dir=VIS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    drawn = False
    for ax, opt in zip(axes, ("adamw", "sgd")):
        for tok in ("char", "word", "bpe"):
            p = Path(outputs_dir) / f"{opt}_{tok}" / "history.csv"
            if not p.exists() or p.stat().st_size == 0:
                continue
            h = pd.read_csv(p)
            if h.empty:
                continue
            ax.plot(h["step"], h["train_loss"], color=_COL[tok], alpha=.5, label=f"{tok} train")
            v = h.dropna(subset=["val_loss"])
            ax.plot(v["step"], v["val_loss"], "o-", color=_COL[tok], label=f"{tok} val")
            drawn = True
        ax.set_title(opt.upper())
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Cross-entropy (nats/token)")
        ax.grid(ls="--", alpha=.4)
        ax.legend(fontsize=8)
    fig.suptitle("Training and validation loss (loss values are token-space)")
    return _save(fig, Path(vis_dir) / "training_curves.png") if drawn else (plt.close(fig) or None)


def plot_best_val(comp, vis_dir=VIS_DIR):
    if comp.empty or comp["best_val_loss"].isna().all():
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(6), comp["best_val_loss"].fillna(0), color=[_color(e) for e in comp["experiment_id"]])
    ax.set_xticks(range(6))
    ax.set_xticklabels(comp["experiment_id"], rotation=20)
    ax.set_ylabel("Best validation loss (nats/token)")
    ax.set_title("Best validation loss (compare within a tokenizer only)")
    return _save(fig, Path(vis_dir) / "best_val_loss.png")


def plot_perplexity(comp, vis_dir=VIS_DIR):
    if comp.empty or comp["test_perplexity"].isna().all():
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, tok in zip(axes, ("char", "word", "bpe")):
        sub = comp[comp["tokenizer"] == tok]
        ax.bar(sub["optimizer"], sub["test_perplexity"].fillna(0), color=_COL[tok])
        ax.set_title(f"{tok} tokens")
        ax.set_ylabel("Test perplexity")
    fig.suptitle("Test perplexity per tokenizer (NOT comparable across panels)")
    return _save(fig, Path(vis_dir) / "perplexity_by_tokenizer.png")


def plot_human(human_summary, vis_dir=VIS_DIR):
    if human_summary is None or human_summary.empty or human_summary["n_ratings"].sum() == 0:
        return None
    hs = human_summary.set_index("experiment_id").reindex(EXPERIMENT_IDS)
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.8 / 6
    for i, eid in enumerate(EXPERIMENT_IDS):
        vals = [hs.loc[eid, f"{c}_mean"] for c in CRITERIA]
        ax.bar(np.arange(len(CRITERIA)) + i * w, np.nan_to_num(vals), w, label=eid)
    ax.set_xticks(np.arange(len(CRITERIA)) + 2.5 * w)
    ax.set_xticklabels(CRITERIA)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_title("Human evaluation by criterion")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    p1 = _save(fig, Path(vis_dir) / "human_criteria.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(6), np.nan_to_num(hs["average_score_mean"]),
           yerr=np.nan_to_num(hs["average_score_std"]), capsize=4,
           color=[_color(e) for e in EXPERIMENT_IDS])
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{e}\n(n={int(n)})" for e, n in zip(EXPERIMENT_IDS, hs["n_ratings"])])
    ax.set_ylim(0, 5)
    ax.set_ylabel("Overall human score (1-5)")
    ax.set_title("Overall human score (mean ± SD of ratings; small sample)")
    return [p1, _save(fig, Path(vis_dir) / "human_overall.png")]