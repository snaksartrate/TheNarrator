"""Entry point. Run from the repository root.

  python main.py --smoke                 # all six, tiny, separate folder
  python main.py --smoke --exp sgd_bpe   # one smoke run
  python main.py --exp adamw_char        # one real experiment (add --resume to continue)
  python main.py --all                   # all six real experiments, sequentially
Optional overrides (apply to every run): --max-steps N --batch-size N --eval-every N
"""
import argparse
import json
from pathlib import Path

import config
from src.data import StoryDataset, load_split
from src.generator import GenConfig, generate_and_save
from src.model import ModelConfig, TransformerLM
from src.trainer import TrainConfig, build_token_stream, set_seed, train
from tokenizers import load_tokenizer


def load_data(tok_name, settings, cache):
    """Load the saved tokenizer and train/validation windows (test split is never loaded)."""
    if tok_name not in cache:
        tok = load_tokenizer(config.TOKENIZER_FILES[tok_name])
        block = config.SHARED_MODEL["block_size"]
        sets = []
        for split in ("train", "validation"):
            stories = load_split(split)
            if settings["max_stories"]:
                stories = stories[:settings["max_stories"]]
            sets.append(StoryDataset(build_token_stream(stories, tok), block))
        cache[tok_name] = (tok, *sets)
    return cache[tok_name]


def tokenizer_unk_rate(tok_name):
    f = Path(config.TOKENIZER_DIR) / "tokenizer_stats.json"
    if f.exists():
        for row in json.loads(f.read_text(encoding="utf-8")):
            if row["tokenizer"] == tok_name:
                return row.get("val_unk_rate")
    return None


def run_experiment(exp_id, settings, resume=False, cache=None):
    opt_name, tok_name = config.EXPERIMENTS[exp_id]
    tok, train_ds, val_ds = load_data(tok_name, settings, cache if cache is not None else {})
    opt = config.OPTIMIZERS[opt_name]
    cfg = TrainConfig(experiment_id=exp_id, optimizer=opt_name, lr=opt["lr"],
                      momentum=opt.get("momentum"), out_dir=str(settings["out_dir"]),
                      **settings["train"])

    set_seed(cfg.seed)
    model = TransformerLM(ModelConfig(vocab_size=tok.vocab_size, **config.SHARED_MODEL))
    assert model.cfg.vocab_size == tok.vocab_size

    info = {"name": tok_name, "path": config.TOKENIZER_FILES[tok_name],
            "vocab_size": tok.vocab_size}
    extra = {"smoke_test": settings["smoke"], "dataset": config.DATASET_NOTE,
             "val_unk_rate": tokenizer_unk_rate(tok_name),
             "limitation": "same block_size in tokens covers different raw text per tokenizer"}
    meta = train(model, train_ds, val_ds, cfg, info, extra_meta=extra,
                 resume=resume, overwrite=settings["smoke"])

    run_dir = Path(settings["out_dir"]) / exp_id
    n_gen = 0
    if (run_dir / "best.pt").exists():
        g = generate_and_save(run_dir, GenConfig(**settings["gen"]),
                              n_stories=settings["n_stories"], device=cfg.device)
        n_gen = len(g["stories"])
    else:
        print(f"[{exp_id}] no checkpoint (diverged before first validation), skipping generation")
    return {"experiment_id": exp_id, "status": meta["status"],
            "steps": meta["steps_completed"], "best_val_loss": meta["best_val_loss"],
            "sec_per_100_steps": meta["seconds_per_100_steps"], "stories": n_gen}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--exp", choices=list(config.EXPERIMENTS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-every", type=int)
    a = p.parse_args()

    if a.exp:
        ids = [a.exp]
    elif a.all or a.smoke:
        ids = list(config.EXPERIMENTS)
    else:
        p.error("choose --smoke, --exp ID or --all")

    over = {k: v for k, v in dict(max_steps=a.max_steps, batch_size=a.batch_size,
                                  eval_every=a.eval_every).items() if v is not None}
    settings = config.get_settings(smoke=a.smoke, **over)
    out = Path(settings["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        (out / "SMOKE_TEST_NOT_RESULTS.txt").write_text(
            "Smoke test outputs. Not research results.\n", encoding="utf-8")

    cache, rows = {}, []
    for exp_id in ids:
        try:
            rows.append(run_experiment(exp_id, settings, a.resume, cache))
        except FileExistsError as e:
            print(f"[{exp_id}] skipped: {e}")
            rows.append({"experiment_id": exp_id, "status": "skipped (exists, use --resume)"})

    print("\n{:<12} {:<12} {:>6} {:>10} {:>10} {:>8}".format(
        "experiment", "status", "steps", "best_val", "s/100step", "stories"))
    for r in rows:
        bv, sp = r.get("best_val_loss"), r.get("sec_per_100_steps")
        print("{:<12} {:<12} {:>6} {:>10} {:>10} {:>8}".format(
            r["experiment_id"], r["status"][:12], r.get("steps", "-"),
            f"{bv:.4f}" if bv is not None else "-",
            f"{sp:.1f}" if sp is not None else "-", r.get("stories", "-")))
    if len(ids) > 1:
        (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()