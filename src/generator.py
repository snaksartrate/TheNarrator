"""Story generation for TheNarrator (same settings for all six models)."""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch

from src.trainer import build_model_from_checkpoint, load_checkpoint, resolve_device


@dataclass
class GenConfig:
    max_new_tokens: int = 300
    temperature: float = 1.0
    top_k: Optional[int] = None
    greedy: bool = False
    seed: int = 42

    def validate(self, vocab_size):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and not 0 < self.top_k <= vocab_size:
            raise ValueError(f"top_k must be in [1, {vocab_size}]")


def _next_token(logits, gen, rng):
    if gen.greedy:
        return int(logits.argmax(dim=-1))
    logits = logits / gen.temperature
    if gen.top_k is not None:
        kth = torch.topk(logits, gen.top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        raise RuntimeError("Invalid probability distribution (NaN/Inf) during sampling")
    return int(torch.multinomial(probs, 1, generator=rng))


@torch.no_grad()
def generate_ids(model, tokenizer, gen, device="cpu"):
    """Start at BOS, stop at EOS or max_new_tokens. Returns ids without BOS/EOS."""
    gen.validate(model.cfg.vocab_size)
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(gen.seed)
    ids = [tokenizer.bos_id]
    for _ in range(gen.max_new_tokens):
        ctx = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits = model(ctx)[0][:, -1, :]          # crop to block_size, last position
        nxt = _next_token(logits, gen, rng)
        if nxt == tokenizer.eos_id:
            break
        ids.append(nxt)
    return ids[1:]


def generate_story(model, tokenizer, gen, device="cpu"):
    """Returns (text, num_tokens, seconds)."""
    t0 = time.perf_counter()
    ids = generate_ids(model, tokenizer, gen, device)
    text = tokenizer.decode(ids)
    return text, len(ids), time.perf_counter() - t0


def load_run(run_dir, checkpoint="best.pt", device="cpu"):
    """Load model + the matching tokenizer from <run_dir>/<checkpoint>."""
    from tokenizers import load_tokenizer
    path = Path(run_dir) / checkpoint
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = load_checkpoint(path, device)
    model = build_model_from_checkpoint(ckpt, device).eval()
    tokenizer = load_tokenizer(ckpt["tokenizer"]["path"])
    if tokenizer.vocab_size != model.cfg.vocab_size:
        raise ValueError("Tokenizer vocab size does not match the checkpoint")
    return model, tokenizer, ckpt


def generate_and_save(run_dir, gen, n_stories=3, checkpoint="best.pt",
                      device="cpu", subdir="generations"):
    """Write story_00.txt ... and generation_meta.json inside the run's own folder."""
    device = resolve_device(device)
    model, tokenizer, ckpt = load_run(run_dir, checkpoint, device)
    out = Path(run_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    stories = []
    for i in range(n_stories):
        g = GenConfig(**{**asdict(gen), "seed": gen.seed + i})   # story i uses seed+i
        text, n_tok, secs = generate_story(model, tokenizer, g, device)
        name = f"story_{i:02d}.txt"
        (out / name).write_text(text, encoding="utf-8")
        stories.append({"file": name, "seed": g.seed, "num_tokens": n_tok, "seconds": secs})
    meta = {
        "experiment_id": ckpt["experiment_id"],
        "tokenizer": ckpt["tokenizer"]["name"],
        "optimizer": ckpt["optimizer_settings"]["name"],
        "checkpoint": checkpoint, "checkpoint_step": ckpt["step"],
        "generation_settings": asdict(gen), "stories": stories,
        "total_seconds": sum(s["seconds"] for s in stories),
    }
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta