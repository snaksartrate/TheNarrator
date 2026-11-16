"""Checks the README acceptance criteria. Run: python -m src.data.validate"""
import json
import tempfile
from pathlib import Path

from .constants import PROCESSED_DIR, RAW_FILE
from .dataset import load_split
from .preprocess import read_raw, run_pipeline


def main():
    splits = {n: load_split(n) for n in ("train", "validation", "test")}

    # 1. no duplicates anywhere
    all_stories = [s for items in splits.values() for s in items]
    keys = [" ".join(s.lower().split()) for s in all_stories]
    assert len(keys) == len(set(keys)), "FAIL: duplicate stories found"
    print("OK  no duplicate stories")

    # 2. no overlap between splits
    a, b, c = (set(splits[n]) for n in ("train", "validation", "test"))
    assert not (a & b or a & c or b & c), "FAIL: splits overlap"
    print("OK  train/validation/test do not overlap")

    # 3. stats file matches the data
    stats = json.load(open(PROCESSED_DIR / "dataset_stats.json"))
    for n, items in splits.items():
        assert stats["splits"][n]["num_stories"] == len(items), f"FAIL: stats mismatch ({n})"
    print("OK  dataset_stats.json matches the split files")

    # 4. reproducible: re-running from the raw file gives identical files
    with tempfile.TemporaryDirectory() as tmp:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(read_raw(RAW_FILE), out_dir=tmp)
        for n in ("train", "validation", "test"):
            same = (Path(tmp) / f"{n}.txt").read_bytes() == (PROCESSED_DIR / f"{n}.txt").read_bytes()
            assert same, f"FAIL: {n}.txt is not reproducible"
    print("OK  re-running gives byte-identical splits")
    print("All checks passed.")


if __name__ == "__main__":
    main()
