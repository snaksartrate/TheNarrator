"""
Fit all three tokenizers on the training split, save them, and report stats.

Run from the repository root:

    python -m tokenizers.train_tokenizers
    python -m tokenizers.train_tokenizers --bpe-vocab 4000 --min-freq 3

If dataset/processed/train.txt does not exist yet (dataset module still in
progress), a small built in sample is used so the script can be demoed.

"Training" here means building the vocabulary from the TRAIN split only.
Validation and test text are never used to fit a tokenizer (no data leakage);
they are only used to measure the unknown token rate.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from . import get_tokenizer

DEMO_STORY = (
    "Once upon a time there was a little fox who lived near a quiet river.\n"
    "Every morning the fox walked to the water and watched the fish swim by.\n"
    "\"Why do you never sleep?\" the fox asked a tired old turtle.\n"
    "The turtle smiled slowly. \"Because the river is always telling stories,\" he said.\n"
    "The fox sat down, and together they listened until the sun went down."
)
DEMO_TEXT = "<|endoftext|>".join([DEMO_STORY] * 20)
DEMO_VAL = (
    "One evening the fox found a shiny stone by the river.\n"
    "\"What a strange stone,\" the turtle said, \"it must hold a story!\""
)


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Train and evaluate the three tokenizers")
    p.add_argument("--train", default="dataset/processed/train.txt")
    p.add_argument("--val", default="dataset/processed/validation.txt")
    p.add_argument("--out", default="tokenizers/saved", help="folder for saved vocab files")
    p.add_argument("--bpe-vocab", type=int, default=4000, help="target BPE vocabulary size")
    p.add_argument("--min-freq", type=int, default=2, help="min word frequency for word tokenizer")
    p.add_argument("--story-separator", default="<|endoftext|>",
                   help="marker between stories in the text files (removed before fitting)")
    p.add_argument("--max-chars", type=int, default=None,
                   help="use only the first N characters of train text (faster BPE)")
    args = p.parse_args()

    if Path(args.train).exists():
        train_text = read_text(args.train)
        print(f"Train text: {args.train}")
    else:
        train_text = DEMO_TEXT
        print(f"{args.train} not found, using built in demo text.")

    if Path(args.val).exists():
        val_text = read_text(args.val)
        print(f"Validation text: {args.val}")
    else:
        val_text = DEMO_VAL
        print(f"{args.val} not found, using built in demo validation text.")

    if args.max_chars:
        train_text = train_text[: args.max_chars]

    # split into stories, so the separator marker itself is never tokenized
    sep = args.story_separator
    if sep and sep in train_text:
        stories = [x.strip() for x in train_text.split(sep) if x.strip()]
        train_text = "\n\n".join(stories)
        val_text = "\n\n".join(x.strip() for x in val_text.split(sep) if x.strip())
    else:
        stories = None
        print(f"Separator {sep!r} not found, per story stats will be skipped.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "char": {},
        "word": {"min_freq": args.min_freq},
        "bpe": {"vocab_size": args.bpe_vocab},
    }

    rows = []
    for name, kwargs in configs.items():
        tok = get_tokenizer(name, **kwargs)

        t0 = time.time()
        tok.train(train_text)
        train_time = time.time() - t0

        t0 = time.time()
        train_stats = tok.get_stats(train_text)
        encode_time = time.time() - t0
        val_stats = tok.get_stats(val_text)

        tok.save(out_dir / f"{name}_tokenizer.json")

        if stories:
            lengths = [len(tok.encode(x)) for x in stories]
            avg_len = round(sum(lengths) / len(lengths), 1)
            max_len = max(lengths)
        else:
            avg_len = max_len = None

        rows.append({
            "tokenizer": name,
            "vocab_size": tok.vocab_size,
            "train_tokens": train_stats["num_tokens"],
            "avg_tokens_per_story": avg_len,
            "max_tokens_per_story": max_len,
            "chars_per_token": round(train_stats["chars_per_token"], 3),
            "val_unk_rate": round(val_stats["unk_rate"], 5),
            "fit_seconds": round(train_time, 2),
            "encode_seconds": round(encode_time, 2),
        })

    # console table
    headers = list(rows[0].keys())
    print()
    print(" | ".join(f"{h[:12]:>12}" for h in headers))
    for r in rows:
        print(" | ".join(f"{str(r[h]):>12}" for h in headers))

    # machine readable copies for the report
    with open(out_dir / "tokenizer_stats.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "tokenizer_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    # side by side demo on one sentence
    sample = val_text.strip().split("\n")[0]
    print(f"\nSample: {sample!r}")
    for name in configs:
        from . import load_tokenizer
        tok = load_tokenizer(out_dir / f"{name}_tokenizer.json")
        ids = tok.encode(sample)
        pieces = [tok.id_to_token[i] for i in ids]
        print(f"\n{name}: {len(ids)} tokens")
        print(pieces[:40])
        print("round trip:", repr(tok.decode(ids)))

    print(f"\nSaved tokenizers and stats to {out_dir}/")


if __name__ == "__main__":
    main()
