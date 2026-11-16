"""Shared constants for the dataset module."""
from pathlib import Path

# Repository root (this file lives in <root>/src/data/)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "raw"
PROCESSED_DIR = ROOT / "dataset" / "processed"
RAW_FILE = RAW_DIR / "tinystories_subset.jsonl"

# Line that separates one story from the next inside train/validation/test.txt
STORY_SEP = "<|endofstory|>"

SEED = 42                      # fixed seed -> reproducible split
SPLIT_RATIOS = (0.80, 0.10, 0.10)   # train, validation, test
NUM_RAW_STORIES = 30000        # how many stories to download
MIN_CHARS = 100                # drop stories shorter than this
