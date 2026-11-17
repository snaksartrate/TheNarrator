"""Entry point. Run from the repository root.

  python main.py --smoke                 # all six, tiny, separate folder
  python main.py --smoke --exp sgd_bpe   # one smoke run
  python main.py --exp adamw_char        # one real experiment (add --resume to continue)
  python main.py --all                   # all six real experiments, sequentially
Optional overrides (apply to every run): --max-steps N --batch-size N --eval-every N
"""
import argparse
import json
import sys
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


def generation_main(a, p, prompt):
    """Prompt-conditioned comparison. Never trains. Always exits."""
    conflicts = [n for n, v in (("--smoke", a.smoke), ("--all", a.all), ("--resume", a.resume),
                                ("--max-steps", a.max_steps), ("--batch-size", a.batch_size),
                                ("--eval-every", a.eval_every)) if v]
    if conflicts:
        p.error(f"prompt generation cannot be combined with training options: {', '.join(conflicts)}")
    if not prompt.strip():
        p.error("the prompt is empty. Give a non-empty story opening.")
    over = {k: v for k, v in dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                                  top_k=a.top_k, seed=a.seed).items() if v is not None}
    gen = GenConfig(**{**config.GEN, **over})
    try:
        gen.validate(10 ** 9)
    except ValueError as e:
        p.error(str(e))
    from src.prompt_comparison import NoCheckpointsError, RunLimitError, run_comparison
    from src.prompt_display import render
    try:
        run = run_comparison(prompt, gen, experiments=[a.exp] if a.exp else None,
                             device=a.device or config.TRAIN["device"],
                             write_report=not a.no_report)
    except (NoCheckpointsError, RunLimitError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(render(run))
    raise SystemExit(1 if run["status"] == "failed" else 0)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="TheNarrator: train the six models, or continue your own story opening.",
        epilog=('Generate (inference only, compares all six trained models):\n'
                '  python main.py "A little girl discovered a mysterious key beneath her bed."\n'
                '  python main.py --generate "..." [--exp ID] [--max-new-tokens N] [--temperature F]\n'
                '                 [--top-k N] [--seed N] [--device DEVICE] [--no-report]\n'
                'Train:\n'
                '  python main.py --smoke | --exp ID [--resume] | --all   '
                '[--max-steps N --batch-size N --eval-every N]'))
    p.add_argument("prompt", nargs="?", help="story opening to continue (generation mode)")
    p.add_argument("--generate", metavar="PROMPT", help="same as the positional prompt")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--exp", choices=list(config.EXPERIMENTS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-every", type=int)
    g = p.add_argument_group("generation options (prompt mode only)")
    g.add_argument("--max-new-tokens", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--seed", type=int)
    g.add_argument("--device")
    g.add_argument("--no-report", action="store_true",
                   help="skip the Markdown report and CSV (prompt, stories, JSON are still saved)")
    a = p.parse_args()

    if a.prompt is not None and a.generate is not None:
        p.error("give the prompt either positionally or with --generate, not both")
    prompt = a.generate if a.generate is not None else a.prompt
    if prompt is not None:
        generation_main(a, p, prompt)
    used = [n for n, v in (("--max-new-tokens", a.max_new_tokens), ("--temperature", a.temperature),
                           ("--top-k", a.top_k), ("--seed", a.seed), ("--device", a.device),
                           ("--no-report", a.no_report)) if v not in (None, False)]
    if used:
        p.error(f"{', '.join(used)} only apply to prompt generation")

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