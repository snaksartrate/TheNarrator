"""Training pipeline for TheNarrator.

Budget: a fixed number of optimizer steps (cfg.max_steps), not epochs.
Each step draws one batch of non-overlapping windows from the train stream.
The batch for step s depends only on (seed, s), so training is reproducible
and can be resumed exactly.

Supports exactly two optimizers: AdamW and SGD with momentum.
Fixed learning rate, no scheduler. Gradient clipping is shared by both.
The test split is never touched here.
"""
import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data import StoryDataset, load_split
from src.model import ModelConfig, TransformerLM

OPTIMIZERS = ("adamw", "sgd")
HISTORY_FIELDS = ["step", "train_loss", "val_loss",
                  "train_seconds", "eval_seconds", "seconds_per_100_steps"]


# --------------------------------------------------------------- config
@dataclass
class TrainConfig:
    experiment_id: str                 # e.g. "adamw_char"
    optimizer: str                     # "adamw" or "sgd"
    lr: float                          # fixed learning rate
    momentum: Optional[float] = 0.9    # SGD only (ignored/None for AdamW)
    grad_clip: Optional[float] = 1.0   # max grad norm, None disables
    batch_size: int = 8
    max_steps: int = 500
    eval_every: int = 100
    log_every: int = 10
    eval_batches: Optional[int] = 50   # validation batches per eval, None = all
    seed: int = 42
    device: str = "cpu"                # "cpu", "cuda" or "auto"
    out_dir: str = "experiments/outputs"

    def __post_init__(self):
        self.optimizer = self.optimizer.lower()
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'. Choose from {OPTIMIZERS}")
        if self.optimizer == "adamw":
            self.momentum = None
        elif self.momentum is None:
            raise ValueError("SGD needs a momentum value")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        for name in ("batch_size", "max_steps", "eval_every", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")
        if self.eval_batches is not None and self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive or None")

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------- helpers
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return dev


def create_optimizer(model, cfg):
    """Return AdamW or SGD(momentum). No scheduler anywhere."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose from {OPTIMIZERS}")


def optimizer_settings(opt):
    """All settings actually in use (includes PyTorch defaults), for metadata."""
    g = opt.param_groups[0]
    s = {"name": type(opt).__name__, "lr": g["lr"], "weight_decay": g["weight_decay"]}
    if isinstance(opt, torch.optim.SGD):
        s["momentum"] = g["momentum"]
    else:
        s["betas"] = list(g["betas"])
    return s


# --------------------------------------------------------------- data
def build_token_stream(stories, tokenizer):
    """Encode each story on its own (BOS ... EOS) and concatenate in order.

    The file delimiter <|endofstory|> was already removed by load_split, so it
    is never encoded. Story boundaries are marked only by EOS then BOS.
    """
    ids = []
    for story in stories:
        ids.extend(tokenizer.encode(story, add_special_tokens=True))
    return ids


def build_split_dataset(split, tokenizer, block_size, processed_dir=None):
    """One split -> one independent token stream -> StoryDataset windows."""
    if processed_dir is None:
        stories = load_split(split)
    else:
        stories = load_split(split, processed_dir)
    return StoryDataset(build_token_stream(stories, tokenizer), block_size)


def get_batch(ds, indices, device):
    pairs = [ds[int(i)] for i in indices]
    x = torch.stack([p[0] for p in pairs]).to(device)
    y = torch.stack([p[1] for p in pairs]).to(device)
    return x, y


class WindowSampler:
    """Reproducible batches: batch for step s depends only on (seed, s).

    Windows are shuffled with a seeded generator and used without
    replacement; after a full pass a new shuffle (seed + pass number) starts.
    """

    def __init__(self, n_windows, batch_size, seed):
        self.n = n_windows
        self.batch_size = batch_size
        self.seed = seed
        self.batches_per_pass = n_windows // batch_size
        if self.batches_per_pass == 0:
            raise ValueError(f"Only {n_windows} windows, fewer than batch_size {batch_size}")
        self._pass = None
        self._perm = None

    def batch_indices(self, step):  # step is 0-based
        p, pos = divmod(step, self.batches_per_pass)
        if p != self._pass:
            g = torch.Generator()
            g.manual_seed(self.seed + p)
            self._perm = torch.randperm(self.n, generator=g)
            self._pass = p
        return self._perm[pos * self.batch_size:(pos + 1) * self.batch_size].tolist()


def eval_indices(n_windows, eval_batches, batch_size, seed):
    """Fixed validation windows, identical at every evaluation."""
    if eval_batches is None or eval_batches * batch_size >= n_windows:
        return list(range(n_windows))
    return sorted(random.Random(seed).sample(range(n_windows), eval_batches * batch_size))


@torch.no_grad()
def evaluate(model, ds, indices, batch_size, device):
    """Mean next-token cross-entropy over the given windows (no gradients)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(indices), batch_size):
        x, y = get_batch(ds, indices[i:i + batch_size], device)
        _, loss = model(x, y)
        total += loss.item() * x.size(0)   # all windows have equal length
        count += x.size(0)
    if was_training:
        model.train()
    return total / count


# --------------------------------------------------------------- checkpoints
def save_checkpoint(path, state):
    path = str(path)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)        # atomic: never a half-written file


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def build_model_from_checkpoint(ckpt, device="cpu"):
    """Rebuild the exact architecture and load weights strictly."""
    model = TransformerLM(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device)


def write_history_csv(history, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        for row in history:
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in HISTORY_FIELDS})


def _check_resume(ckpt, model, cfg, tokenizer_info, opt_settings, n_train):
    problems = []

    def chk(name, saved, now):
        if saved != now:
            problems.append(f"{name}: checkpoint={saved!r}, now={now!r}")

    chk("experiment_id", ckpt["experiment_id"], cfg.experiment_id)
    chk("model_config", ckpt["model_config"], model.cfg.to_dict())
    chk("optimizer_settings", ckpt["optimizer_settings"], opt_settings)
    chk("seed", ckpt["seed"], cfg.seed)
    chk("batch_size", ckpt["train_config"]["batch_size"], cfg.batch_size)
    chk("train_windows", ckpt["train_windows"], n_train)
    keys = ("name", "vocab_size")
    chk("tokenizer", {k: ckpt["tokenizer"].get(k) for k in keys},
        {k: tokenizer_info.get(k) for k in keys})
    if problems:
        raise ValueError("Cannot resume, run differs from checkpoint:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------- training
def train(model, train_ds, val_ds, cfg, tokenizer_info, extra_meta=None,
          resume=False, overwrite=False, log=print):
    """Train for cfg.max_steps optimizer steps. Returns the run metadata dict.

    tokenizer_info: plain dict, e.g. {"name": "char", "path": "...", "vocab_size": 84}
    extra_meta:     plain dict merged into run_meta.json (e.g. validation UNK rate)
    Call set_seed(cfg.seed) before building the model so initial weights are reproducible.
    """
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = create_optimizer(model, cfg)
    opt_settings = optimizer_settings(optimizer)

    run_dir = Path(cfg.out_dir) / cfg.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    history_path, meta_path = run_dir / "history.csv", run_dir / "run_meta.json"

    if len(val_ds) == 0:
        raise ValueError("Validation set has no windows")
    sampler = WindowSampler(len(train_ds), cfg.batch_size, cfg.seed)
    val_idx = eval_indices(len(val_ds), cfg.eval_batches, cfg.batch_size, cfg.seed)

    step, best_val, best_step = 0, float("inf"), 0
    history, train_secs, eval_secs = [], 0.0, 0.0

    if last_path.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"{last_path} exists. Use resume=True to continue or overwrite=True to start over.")
    if resume and last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        _check_resume(ckpt, model, cfg, tokenizer_info, opt_settings, len(train_ds))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["rng_state"].cpu())
        step, best_val, best_step = ckpt["step"], ckpt["best_val_loss"], ckpt["best_step"]
        history, train_secs, eval_secs = ckpt["history"], ckpt["train_seconds"], ckpt["eval_seconds"]
        log(f"[{cfg.experiment_id}] resumed from step {step}")
    else:
        set_seed(cfg.seed)

    def make_state():
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step, "best_val_loss": best_val, "best_step": best_step,
            "model_config": model.cfg.to_dict(),
            "tokenizer": tokenizer_info,
            "optimizer_settings": opt_settings,
            "train_config": cfg.to_dict(),
            "seed": cfg.seed, "experiment_id": cfg.experiment_id,
            "train_windows": len(train_ds),
            "history": history, "rng_state": torch.get_rng_state(),
            "train_seconds": train_secs, "eval_seconds": eval_secs,
        }

    status = "completed"
    interval_loss, interval_n, interval_secs = 0.0, 0, 0.0
    model.train()
    while step < cfg.max_steps:
        t0 = time.perf_counter()
        x, y = get_batch(train_ds, sampler.batch_indices(step), device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            status = "diverged"
            log(f"[{cfg.experiment_id}] loss is {loss.item()} at step {step + 1}, stopping")
            break
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        dt = time.perf_counter() - t0
        step += 1
        train_secs += dt
        interval_secs += dt
        interval_loss += loss.item()
        interval_n += 1

        do_eval = step % cfg.eval_every == 0 or step == cfg.max_steps
        if step % cfg.log_every == 0 or do_eval:
            row = {"step": step, "train_loss": interval_loss / interval_n, "val_loss": None,
                   "train_seconds": train_secs, "eval_seconds": eval_secs,
                   "seconds_per_100_steps": interval_secs / interval_n * 100}
            interval_loss, interval_n, interval_secs = 0.0, 0, 0.0
            if do_eval:
                t1 = time.perf_counter()
                val = evaluate(model, val_ds, val_idx, cfg.batch_size, device)
                eval_secs += time.perf_counter() - t1
                row["eval_seconds"] = eval_secs
                if not np.isfinite(val):
                    status = "diverged"
                    history.append(row)
                    log(f"[{cfg.experiment_id}] validation loss is {val} at step {step}, stopping")
                    break
                row["val_loss"] = val
                is_best = val < best_val
                if is_best:
                    best_val, best_step = val, step
                history.append(row)
                save_checkpoint(last_path, make_state())
                if is_best:
                    save_checkpoint(best_path, make_state())
                write_history_csv(history, history_path)
            else:
                history.append(row)
            val_txt = "" if row["val_loss"] is None else f" val {row['val_loss']:.4f}"
            log(f"[{cfg.experiment_id}] step {step}/{cfg.max_steps} "
                f"train {row['train_loss']:.4f}{val_txt} "
                f"{row['seconds_per_100_steps']:.1f}s/100 steps")

    write_history_csv(history, history_path)
    n_windows = len(train_ds)
    drawn = step * cfg.batch_size
    meta = {
        "experiment_id": cfg.experiment_id, "status": status,
        "steps_completed": step, "max_steps": cfg.max_steps,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "best_step": best_step,
        "final_val_loss": next((r["val_loss"] for r in reversed(history)
                                if r.get("val_loss") is not None), None),
        "train_seconds": train_secs, "eval_seconds": eval_secs,
        "seconds_per_100_steps": (train_secs / step * 100) if step else None,
        "windows_drawn": drawn, "train_windows_total": n_windows,
        "epoch_fraction": drawn / n_windows,
        "tokens_processed": drawn * model.cfg.block_size,
        "batch_size": cfg.batch_size, "block_size": model.cfg.block_size,
        "num_parameters": model.num_parameters(),
        "model_config": model.cfg.to_dict(), "tokenizer": tokenizer_info,
        "optimizer_settings": opt_settings, "train_config": cfg.to_dict(),
        "seed": cfg.seed, "device": str(device), "torch_version": torch.__version__,
        "budget": "fixed optimizer steps (not epochs)",
        "checkpoint_last": str(last_path) if last_path.exists() else None,
        "checkpoint_best": str(best_path) if best_path.exists() else None,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta