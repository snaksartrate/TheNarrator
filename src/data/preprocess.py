"""Download, clean, deduplicate and split the story dataset.

Run from the repository root:
    python -m src.data.preprocess
"""
import argparse
import json
import random
import re
import unicodedata
from itertools import islice
from pathlib import Path

from .constants import (MIN_CHARS, NUM_RAW_STORIES, PROCESSED_DIR, RAW_DIR,
                        RAW_FILE, SEED, SPLIT_RATIOS, STORY_SEP)

REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
}


# ---------------------------------------------------------------- download
def download_raw(num_stories=NUM_RAW_STORIES, out_file=RAW_FILE):
    """Stream the first `num_stories` TinyStories and save them as JSONL."""
    from datasets import load_dataset  # imported here so other steps don't need it

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for row in islice(ds, num_stories):
            f.write(json.dumps({"text": row["text"]}) + "\n")
    print(f"Saved {num_stories} raw stories to {out_file}")


def read_raw(raw_file=RAW_FILE):
    with open(raw_file, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


# ---------------------------------------------------------------- cleaning
def normalize_text(text):
    """Make the characters and whitespace of one story consistent."""
    text = unicodedata.normalize("NFKD", text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # keep only newline + printable ASCII (keeps the character vocab small)
    text = "".join(ch for ch in text if ch == "\n" or 32 <= ord(ch) < 127)
    text = re.sub(r"[ \t]+", " ", text)      # many spaces -> one
    text = re.sub(r" ?\n ?", "\n", text)     # no spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)   # at most one blank line
    return text.strip()


def clean_stories(raw_stories):
    """Normalize, drop bad samples, remove duplicates. Returns (stories, report)."""
    report = {"raw": len(raw_stories)}
    cleaned = [normalize_text(s) for s in raw_stories if isinstance(s, str)]

    report["removed_empty"] = sum(1 for s in cleaned if not s)
    cleaned = [s for s in cleaned if s]

    report["removed_too_short"] = sum(1 for s in cleaned if len(s) < MIN_CHARS)
    cleaned = [s for s in cleaned if len(s) >= MIN_CHARS]

    report["removed_contains_separator"] = sum(1 for s in cleaned if STORY_SEP in s)
    cleaned = [s for s in cleaned if STORY_SEP not in s]

    seen, unique = set(), []
    for s in cleaned:
        key = " ".join(s.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(s)
    report["removed_duplicates"] = len(cleaned) - len(unique)
    report["final"] = len(unique)
    return unique, report


# ---------------------------------------------------------------- splitting
def split_stories(stories, seed=SEED, ratios=SPLIT_RATIOS):
    """Shuffle with a fixed seed, then split by whole stories."""
    stories = list(stories)
    random.Random(seed).shuffle(stories)
    n = len(stories)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = stories[:n_train]
    val = stories[n_train:n_train + n_val]
    test = stories[n_train + n_val:]
    assert not (set(train) & set(val)) and not (set(train) & set(test)) \
        and not (set(val) & set(test)), "splits overlap!"
    return {"train": train, "validation": val, "test": test}


def write_split(stories, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in stories:
            f.write(s + "\n" + STORY_SEP + "\n")


# ---------------------------------------------------------------- statistics
def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def compute_stats(splits, report):
    stats = {"seed": SEED, "source": "roneneldan/TinyStories (first stories of the train split)",
             "story_separator": STORY_SEP, "cleaning_report": report, "splits": {}}
    for name, stories in splits.items():
        text = "\n".join(stories)
        words = text.split()
        stats["splits"][name] = {
            "num_stories": len(stories),
            "num_characters": len(text),
            "num_words": len(words),
            "avg_words_per_story": round(len(words) / max(len(stories), 1), 1),
            "avg_chars_per_story": round(len(text) / max(len(stories), 1), 1),
            "unique_characters": len(set(text)),
        }
    train_text = "\n".join(splits["train"])
    stats["train_word_vocab_size"] = len(set(_tokens(train_text)))
    stats["train_character_vocab_size"] = len(set(train_text))
    return stats


# ---------------------------------------------------------------- pipeline
def run_pipeline(raw_stories, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stories, report = clean_stories(raw_stories)
    splits = split_stories(stories)
    for name, items in splits.items():
        write_split(items, out_dir / f"{name}.txt")
    stats = compute_stats(splits, report)
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse dataset/raw/ instead of downloading again")
    args = parser.parse_args()
    if not args.skip_download or not RAW_FILE.exists():
        download_raw()
    run_pipeline(read_raw())


if __name__ == "__main__":
    main()
