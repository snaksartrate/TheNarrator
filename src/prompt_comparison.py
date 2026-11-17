"""Prompt-conditioned comparison of the six trained models. Inference only:
never trains, never writes outside experiments/prompt_comparisons/run_NNNNN/."""
import csv
import gc
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import config
from evaluation.metrics import story_metrics
from src.generator import generate_continuation, load_run
from src.trainer import resolve_device

MAX_RUN_ID = 99999
_RUN_RE = re.compile(r"^run_(\d{5})$")
CSV_COLUMNS = ["run_id", "experiment_id", "optimizer", "tokenizer", "status", "error",
               "checkpoint_step", "vocab_size", "prompt_tokens_original", "prompt_tokens_used",
               "truncated", "generated_tokens", "word_count", "distinct_1", "distinct_2",
               "repetition_rate", "lexical_diversity", "load_seconds", "generation_seconds",
               "total_seconds", "words_per_second"]


class RunLimitError(RuntimeError):
    pass


class NoCheckpointsError(RuntimeError):
    pass


# ------------------------------------------------------------ run ids
def allocate_run(root):
    """Reserve the next run_NNNNN directory atomically. Never reuses or overwrites."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        ids = [int(m.group(1)) for p in root.iterdir()
               if p.is_dir() and (m := _RUN_RE.match(p.name)) and int(m.group(1)) >= 1]
        nxt = max(ids, default=0) + 1
        if nxt > MAX_RUN_ID:
            raise RunLimitError(f"The {MAX_RUN_ID}-run limit has been reached in {root}. "
                                "No new comparison was started.")
        d = root / f"run_{nxt:05d}"
        try:
            d.mkdir()                      # atomic: fails if another process won the race
            return nxt, d
        except FileExistsError:
            continue


def _write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ one model
def _run_one(eid, prompt, gen, outputs_dir, checkpoint, device, loader):
    opt, tok_name = config.EXPERIMENTS[eid]
    r = {"experiment_id": eid, "optimizer": opt, "tokenizer": tok_name, "status": "ok",
         "error": "", "checkpoint": checkpoint, "checkpoint_step": None, "vocab_size": None,
         "prompt_tokens_original": None, "prompt_tokens_used": None, "truncated": None,
         "continuation": None, "generated_tokens": None, "load_seconds": None,
         "generation_seconds": None, "total_seconds": None, "words_per_second": None,
         "metrics": None}
    run = Path(outputs_dir) / eid
    if not (run / checkpoint).exists():
        r.update(status="unavailable", error=f"checkpoint not found: {run / checkpoint}")
        return r
    model = None
    try:
        t0 = time.perf_counter()
        model, tok, ckpt = loader(run, checkpoint, device)
        r["load_seconds"] = time.perf_counter() - t0
        if ckpt["experiment_id"] != eid or ckpt["tokenizer"]["name"] != tok_name:
            raise ValueError(f"checkpoint is {ckpt['experiment_id']}/{ckpt['tokenizer']['name']}, "
                             f"expected {eid}/{tok_name}")
        r["checkpoint_step"] = ckpt.get("step")
        r["vocab_size"] = tok.vocab_size
        out = generate_continuation(model, tok, gen, prompt, device)
        text = out["text"]
        r.update(continuation=text, generated_tokens=out["num_tokens"],
                 generation_seconds=out["generation_seconds"], total_seconds=out["total_seconds"],
                 prompt_tokens_original=out["prompt_tokens_original"],
                 prompt_tokens_used=out["prompt_tokens_used"], truncated=out["truncated"])
        if not text.strip():
            r["status"] = "empty"                      # real output, no metrics invented
        else:
            m = story_metrics(text)
            r["metrics"] = m
            if out["generation_seconds"] > 0 and m["word_count"] > 0:
                r["words_per_second"] = m["word_count"] / out["generation_seconds"]
    except Exception as e:                             # one bad model must not stop the rest
        r.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        model = None
        gc.collect()
    return r


# ------------------------------------------------------------ artifacts
def _fmt(v, p=3):
    return "" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def _csv_row(run_id, r):
    m = r["metrics"] or {}
    row = {"run_id": f"{run_id:05d}", **{k: r.get(k) for k in CSV_COLUMNS if k in r}}
    row["generated_tokens"] = r["generated_tokens"]
    for k in ("word_count", "distinct_1", "distinct_2", "repetition_rate", "lexical_diversity"):
        row[k] = m.get(k)
    return {k: ("" if row.get(k) is None else row[k]) for k in CSV_COLUMNS}


def _report_md(run_id, meta, results, prompt):
    g = meta["generation"]
    L = [f"# Prompt comparison - run {run_id:05d}", "",
         f"Timestamp (UTC): {meta['timestamp']}", f"Mode: {meta['mode']}", f"Run status: {meta['status']}",
         "", "## Prompt", ""] + ["> " + x for x in prompt.split("\n")] + [
         "", "## Generation configuration", "",
         f"- max new tokens: {g['max_new_tokens']}", f"- temperature: {g['temperature']}",
         f"- top-k: {g['top_k']}", f"- mode: {'greedy' if g['greedy'] else 'sampling'}",
         f"- seed: {g['seed']} (same integer for every model; it does not give equivalent "
         "samples across different tokenizers)", f"- device: {meta['device']}",
         "", "## Model status", "", "| Experiment | Status | Detail |", "|---|---|---|"]
    for r in results:
        L.append(f"| {r['experiment_id']} | {r['status']} | {r['error'].replace('|', '/')} |")
    L += ["", "## Continuations", ""]
    for r in results:
        L += [f"### {r['experiment_id']} ({r['status']})", ""]
        if r["status"] in ("ok", "empty"):
            L += ["> " + x for x in (r["continuation"] or "(empty continuation)").split("\n")]
        else:
            L.append(f"No continuation: {r['error']}")
        L.append("")
    L += ["## Comparison table", "",
          "| Experiment | Status | Prompt tokens used/original | New tokens | Words | Distinct-1 | "
          "Distinct-2 | Repetition (trigram) | Lexical div. | Gen. time (s) | Words/s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["metrics"] or {}
        used = "" if r["prompt_tokens_used"] is None else f"{r['prompt_tokens_used']}/{r['prompt_tokens_original']}"
        L.append("| " + " | ".join([r["experiment_id"], r["status"], used, _fmt(r["generated_tokens"]),
                 _fmt(m.get("word_count")), _fmt(m.get("distinct_1")), _fmt(m.get("distinct_2")),
                 _fmt(m.get("repetition_rate")), _fmt(m.get("lexical_diversity")),
                 _fmt(r["generation_seconds"], 2), _fmt(r["words_per_second"], 1)]) + " |")
    L += ["", "## Limitations", "",
          "- Illustrative side-by-side demonstration under one prompt and shared settings. It is NOT "
          "the official evaluation and says nothing statistically about which model is better.",
          "- Distinct-n, repetition and lexical diversity depend on continuation length; read them "
          "with the word count. Metric definitions: evaluation/README.md.",
          "- No perplexity is reported here. Official test perplexity lives in experiments/analysis/ "
          "and is token-space (not comparable across tokenizers).",
          "- Generation time depends on CPU load and caching; model loading time is excluded "
          "(see CSV/JSON for load and total time).",
          "- Prompts with characters or words unseen in training become <unk> for the char/word "
          "tokenizers and lower output quality.",
          "- These continuations are NOT part of the official generated-story set or human evaluation.", ""]
    return "\n".join(L)


def _save(run_dir, run_id, meta, results, prompt, write_report):
    stories = Path(run_dir) / "stories"
    stories.mkdir(exist_ok=True)
    for r in results:
        if r["status"] in ("ok", "empty"):
            _write(stories / f"{r['experiment_id']}_{run_id:05d}.txt", r["continuation"])
    _write_json(Path(run_dir) / f"comparison_{run_id:05d}.json",
                {"run_id": f"{run_id:05d}", "prompt": prompt, "results": results})
    if write_report:
        p = Path(run_dir) / f"comparison_{run_id:05d}.csv"
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for r in results:
                w.writerow(_csv_row(run_id, r))
        os.replace(tmp, p)
        _write(Path(run_dir) / f"report_{run_id:05d}.md", _report_md(run_id, meta, results, prompt))
    _write_json(Path(run_dir) / f"metadata_{run_id:05d}.json", meta)


# ------------------------------------------------------------ public API
def run_comparison(prompt, gen, experiments=None, outputs_dir=config.OUTPUT_DIR,
                   root=config.PROMPT_DIR, device="cpu", checkpoint="best.pt",
                   write_report=True, loader=load_run):
    """Generate one continuation per selected experiment (default: all six)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is empty. Enter a non-empty story opening.")
    if experiments is not None:
        bad = [e for e in experiments if e not in config.EXPERIMENTS]
        if bad:
            raise ValueError(f"unknown experiment id(s): {bad}")
    ids = [e for e in config.EXPERIMENTS if experiments is None or e in experiments]
    if not any((Path(outputs_dir) / e / checkpoint).exists() for e in ids):
        raise NoCheckpointsError(f"No {checkpoint} found for {ids} under {outputs_dir}. "
                                 "Train the models first. No run was created.")
    dev = resolve_device(device)
    run_id, run_dir = allocate_run(root)
    meta = {"run_id": f"{run_id:05d}", "status": "started",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "six-model comparison" if len(ids) == len(config.EXPERIMENTS) else "single/partial selection",
            "experiments_requested": ids, "device": str(dev), "torch_version": torch.__version__,
            "checkpoint": checkpoint,
            "generation": {"max_new_tokens": gen.max_new_tokens, "temperature": gen.temperature,
                           "top_k": gen.top_k, "greedy": gen.greedy, "seed": gen.seed}}
    _write(Path(run_dir) / f"prompt_{run_id:05d}.txt", prompt)       # exact text
    _write_json(Path(run_dir) / f"metadata_{run_id:05d}.json", meta)  # survives a crash
    results = [_run_one(e, prompt, gen, outputs_dir, checkpoint, dev, loader) for e in ids]
    n_ok = sum(r["status"] in ("ok", "empty") for r in results)
    meta["status"] = "complete" if n_ok == len(results) else ("partial" if n_ok else "failed")
    meta["models_succeeded"] = n_ok
    _save(run_dir, run_id, meta, results, prompt, write_report)
    return {"run_id": run_id, "run_dir": run_dir, "status": meta["status"], "prompt": prompt,
            "results": results, "meta": meta}