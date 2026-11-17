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


def prepare_prompt_ids(tokenizer, prompt, block_size):
    """Return (ids, info): [BOS] + encoded prompt, no EOS.

    Policy for long prompts: keep BOS (if the tokenizer has one) and the NEWEST
    tokens that fit in block_size; the oldest prompt tokens are dropped.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt is empty. Enter a non-empty story opening.")
    body = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        body = [int(i) for i in body]
    except (TypeError, ValueError):
        raise ValueError("Tokenizer returned malformed output (expected a list of integer ids)")
    vocab = getattr(tokenizer, "vocab_size", None)
    if vocab is not None and any(i < 0 or i >= vocab for i in body):
        raise ValueError("Tokenizer returned token ids outside its vocabulary")
    unk = getattr(tokenizer, "unk_id", None)
    if not any(i != unk for i in body):
        raise ValueError("Prompt has no usable tokens for this tokenizer (empty or all <unk>)")
    bos = getattr(tokenizer, "bos_id", None)
    prefix = [] if bos is None else [bos]
    keep = block_size - len(prefix)
    original = len(body)
    if original > keep:
        body = body[-keep:]
    info = {"prompt_tokens_original": original, "prompt_tokens_used": len(body),
            "truncated": original > len(body), "bos_prepended": bool(prefix)}
    return prefix + body, info


@torch.no_grad()
def _sample(model, tokenizer, gen, device, ids):
    """Autoregressive loop from the context `ids`. Returns ONLY the new ids."""
    gen.validate(model.cfg.vocab_size)
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(gen.seed)
    ids = list(ids)
    start = len(ids)
    for _ in range(gen.max_new_tokens):
        ctx = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits = model(ctx)[0][:, -1, :]          # crop to block_size, last position
        nxt = _next_token(logits, gen, rng)
        if nxt == tokenizer.eos_id:
            break
        ids.append(nxt)
    return ids[start:]


@torch.no_grad()
def generate_ids(model, tokenizer, gen, device="cpu", prompt=None):
    """prompt=None: start at BOS (original behaviour). Otherwise start from the
    prompt. Stops at EOS or max_new_tokens. Returns new ids only (no BOS/EOS/prompt)."""
    if prompt is None:
        ids = [tokenizer.bos_id]
    else:
        ids, _ = prepare_prompt_ids(tokenizer, prompt, model.cfg.block_size)
    return _sample(model, tokenizer, gen, device, ids)


def generate_story(model, tokenizer, gen, device="cpu", prompt=None):
    """Returns (text, num_tokens, seconds). Text is the continuation only."""
    t0 = time.perf_counter()
    ids = generate_ids(model, tokenizer, gen, device, prompt)
    text = tokenizer.decode(ids)
    return text, len(ids), time.perf_counter() - t0


def generate_continuation(model, tokenizer, gen, prompt, device="cpu"):
    """Prompt-conditioned generation with details for the comparison report.

    generation_seconds = sampling loop only; total_seconds = prompt encoding +
    sampling + decoding. Model loading is NOT included.
    """
    t0 = time.perf_counter()
    ids, info = prepare_prompt_ids(tokenizer, prompt, model.cfg.block_size)
    t1 = time.perf_counter()
    new = _sample(model, tokenizer, gen, device, ids)
    t2 = time.perf_counter()
    text = tokenizer.decode(new)
    t3 = time.perf_counter()
    return {"text": text, "num_tokens": len(new), "generation_seconds": t2 - t1,
            "total_seconds": t3 - t0, **info}


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