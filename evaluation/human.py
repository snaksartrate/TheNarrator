"""Blind human-evaluation protocol: sheet generation, validation, summary."""
import csv
import math
import random
from pathlib import Path

import pandas as pd

from .schemas import CRITERIA, EXPERIMENT_IDS, HUMAN_RAW_COLUMNS, RESULTS_DIR


def make_blind_sheet(outputs_dir, out_dir=RESULTS_DIR, seed=42, force=False):
    """Shuffle every saved story, give blind ids. Raters get blind_stories.csv only;
    blind_key.csv (id -> experiment) stays with the organiser until scoring ends.
    Refuses to overwrite an existing key, or to run with missing models, unless force=True."""
    items, missing = [], []
    for eid in EXPERIMENT_IDS:
        gdir = Path(outputs_dir) / eid / "generations"
        files = sorted(gdir.glob("story_*.txt")) if gdir.exists() else []
        if not files:
            missing.append(eid)
        for f in files:
            items.append((eid, f.stem, f.read_text(encoding="utf-8")))
    if not force:
        if (Path(out_dir) / "blind_key.csv").exists():
            raise FileExistsError("blind_key.csv exists; regenerating changes blind ids. "
                                  "Use --force only if no ratings were collected.")
        if missing:
            raise ValueError(f"no stories for {missing}; wait for all six or use --force")
    random.Random(seed).shuffle(items)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "blind_stories.csv", "w", newline="", encoding="utf-8") as a, \
         open(out_dir / "blind_key.csv", "w", newline="", encoding="utf-8") as b:
        wa, wb = csv.writer(a), csv.writer(b)
        wa.writerow(["blind_story_id", "story_text"])
        wb.writerow(["blind_story_id", "experiment_id", "story_id"])
        for i, (eid, sid, text) in enumerate(items, 1):
            wa.writerow([f"B{i:03d}", text])
            wb.writerow([f"B{i:03d}", eid, sid])
    with open(out_dir / "human_ratings_template.csv", "w", newline="", encoding="utf-8") as t:
        csv.writer(t).writerow(HUMAN_RAW_COLUMNS)
    return len(items)


def validate_rating(rec):
    for k in ("evaluator_id", "blind_story_id"):
        if not str(rec.get(k, "")).strip():
            raise ValueError(f"missing {k}")
    for c in CRITERIA:
        try:
            v = float(rec[c])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"missing/non-numeric {c}")
        if not math.isfinite(v) or not 1 <= v <= 5:
            raise ValueError(f"{c} must be in [1, 5], got {v}")
    return rec


def load_ratings(ratings_csv, key_csv):
    """Validate raw ratings, reject duplicates, join experiment ids. Raw rows kept."""
    raw = pd.read_csv(ratings_csv, dtype={"evaluator_id": str, "blind_story_id": str})
    key = pd.read_csv(key_csv, dtype=str)
    for i, rec in raw.iterrows():
        try:
            validate_rating(rec.to_dict())
        except ValueError as e:
            raise ValueError(f"row {i + 2}: {e}")
    if raw.duplicated(["evaluator_id", "blind_story_id"]).any():
        raise ValueError("duplicate evaluator/story ratings")
    df = raw.merge(key, on="blind_story_id", how="left", validate="many_to_one")
    if df["experiment_id"].isna().any():
        raise ValueError("rating for unknown blind_story_id")
    df["average_score"] = df[CRITERIA].mean(axis=1)
    return df


def summarize_human(df):
    """One row per experiment (all six, even if unrated). Mean and SD per criterion."""
    rows = []
    for eid in EXPERIMENT_IDS:
        sub = df[df["experiment_id"] == eid]
        r = {"experiment_id": eid, "n_ratings": len(sub),
             "n_stories": sub["story_id"].nunique(), "n_evaluators": sub["evaluator_id"].nunique()}
        for c in CRITERIA + ["average_score"]:
            r[f"{c}_mean"] = sub[c].mean() if len(sub) else float("nan")
            r[f"{c}_std"] = sub[c].std(ddof=1) if len(sub) > 1 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def run_human_summary(ratings_csv, key_csv, results_dir=RESULTS_DIR):
    df = load_ratings(ratings_csv, key_csv)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "human_evaluation.csv", index=False)
    s = summarize_human(df)
    s.to_csv(results_dir / "human_summary.csv", index=False)
    return s