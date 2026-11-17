"""Held-out test perplexity (token space; NOT comparable across tokenizers).

Every story is encoded alone with BOS/EOS. Each story is cut into windows
x=ids[s:s+n], y=ids[s+1:s+1+n], step = block_size, so every next-token target
(len(ids)-1 per story) is scored exactly once. Windows do not overlap, so
later windows of a long story lose the earlier context (documented limitation).
Loss is a SUM over targets divided by the total target count (token-weighted).
The test split is loaded only inside evaluate_checkpoint().
"""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

import config


def build_windows(story_ids, block_size):
    """Yield (x, y) lists with len(x)==len(y)>=1; covers all targets once."""
    for ids in story_ids:
        for s in range(0, len(ids) - 1, block_size):
            y = ids[s + 1:s + 1 + block_size]
            yield ids[s:s + len(y)], y


@torch.no_grad()
def sequence_nll(model, story_ids, block_size=None, batch_size=16, device="cpu", unk_id=None):
    """Returns dict(total_nll, token_count, unk_count). Restores training mode."""
    block_size = block_size or model.cfg.block_size
    if block_size > model.cfg.block_size:
        raise ValueError("block_size larger than the model context")
    was_training = model.training
    model.eval()
    by_len = defaultdict(list)
    for x, y in build_windows(story_ids, block_size):
        by_len[len(x)].append((x, y))
    total, count, unk = 0.0, 0, 0
    for length in sorted(by_len):
        items = by_len[length]
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            x = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
            y = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                  reduction="sum").item()
            if not math.isfinite(nll):
                raise FloatingPointError("non-finite loss during perplexity evaluation")
            total += nll
            count += y.numel()
            if unk_id is not None:
                unk += int((y == unk_id).sum())
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("no target tokens evaluated")
    return {"total_nll": total, "token_count": count, "unk_count": unk}


def summarize(res):
    mean = res["total_nll"] / res["token_count"]
    try:
        ppl = math.exp(mean)
    except OverflowError:
        raise FloatingPointError(f"perplexity overflow (mean CE {mean})")
    return {
        "test_mean_cross_entropy": mean, "test_perplexity": ppl,
        "test_bits_per_token": mean / math.log(2),
        "test_token_count": res["token_count"],
        "test_unknown_token_count": res["unk_count"],
        "test_unknown_token_rate": res["unk_count"] / res["token_count"],
    }


def evaluate_checkpoint(run_dir, checkpoint="best.pt", device="cpu", batch_size=16,
                        processed_dir=None, max_stories=None):
    """Final-evaluation entry point (only place the test split is read)."""
    from pathlib import Path
    from src.data import load_split
    from src.generator import load_run

    run_dir = Path(run_dir)
    import json
    mp = run_dir / "run_meta.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")).get("smoke_test"):
        raise ValueError(f"{run_dir.name}: smoke-test run, excluded from results")
    model, tok, ckpt = load_run(run_dir, checkpoint, device)
    eid = ckpt["experiment_id"]
    if eid != run_dir.name or eid not in config.EXPERIMENTS:
        raise ValueError(f"checkpoint experiment_id {eid!r} does not match folder {run_dir.name!r}")
    if ckpt["tokenizer"]["name"] != config.EXPERIMENTS[eid][1]:
        raise ValueError("checkpoint tokenizer does not match the experiment matrix")
    stories = load_split("test") if processed_dir is None else load_split("test", processed_dir)
    if max_stories:
        stories = stories[:max_stories]
    ids = [tok.encode(s, add_special_tokens=True) for s in stories]
    res = sequence_nll(model, ids, batch_size=batch_size, device=device, unk_id=tok.unk_id)
    return {"experiment_id": eid, "tokenizer": ckpt["tokenizer"]["name"],
            "optimizer": ckpt["optimizer_settings"]["name"], "checkpoint": checkpoint,
            "checkpoint_step": ckpt["step"], "vocab_size": tok.vocab_size,
            "test_stories": len(stories), "split": "test", "torch_version": torch.__version__,
            **summarize(res),
            "caveat": "token-space perplexity; compare only within one tokenizer"}