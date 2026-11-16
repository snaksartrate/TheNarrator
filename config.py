"""All experiment settings in one place. Shared by all six runs unless noted."""
from src.data.constants import ROOT, PROCESSED_DIR  # noqa: F401  (re-exported)

TOKENIZER_DIR = "tokenizers/saved"            # relative to repo root
TOKENIZER_FILES = {n: f"{TOKENIZER_DIR}/{n}_tokenizer.json" for n in ("char", "word", "bpe")}
OUTPUT_DIR = ROOT / "experiments" / "outputs"
SMOKE_DIR = ROOT / "experiments" / "smoke_outputs"   # smoke tests never go in OUTPUT_DIR

# Six experiments: id -> (optimizer, tokenizer). Do not add rows.
EXPERIMENTS = {f"{o}_{t}": (o, t) for o in ("adamw", "sgd") for t in ("char", "word", "bpe")}

# Same architecture for every run (vocab_size is added from the tokenizer).
SHARED_MODEL = dict(block_size=256, n_embd=128, n_head=4, n_layer=4, ff_mult=4, dropout=0.1)

# Only the optimizer differs between strategies. Fixed LR, no scheduler.
OPTIMIZERS = {
    "adamw": dict(lr=3e-4),
    "sgd": dict(lr=0.1, momentum=0.9),
}

# Shared training settings. Budget = fixed optimizer steps (raise max_steps after timing).
TRAIN = dict(batch_size=8, max_steps=500, eval_every=100, log_every=10,
             eval_batches=50, grad_clip=1.0, seed=42, device="cpu")

# Same generation settings for all six models.
GEN = dict(max_new_tokens=300, temperature=0.8, top_k=40, greedy=False, seed=42)
N_STORIES = 5

# Smoke test: tiny, separate folder, NOT research results.
SMOKE_TRAIN = dict(batch_size=4, max_steps=4, eval_every=2, log_every=1, eval_batches=2)
SMOKE_MAX_STORIES = 60
SMOKE_MAX_NEW_TOKENS = 30

DATASET_NOTE = "roneneldan/TinyStories first 30,000 stories, seed 42, 80/10/10 split"


def get_settings(smoke=False, **overrides):
    """Return the settings for a run. Overrides (e.g. max_steps) apply to all runs."""
    train, gen = dict(TRAIN), dict(GEN)
    n_stories, max_stories, out_dir = N_STORIES, None, OUTPUT_DIR
    if smoke:
        train.update(SMOKE_TRAIN)
        gen["max_new_tokens"] = SMOKE_MAX_NEW_TOKENS
        n_stories, max_stories, out_dir = 1, SMOKE_MAX_STORIES, SMOKE_DIR
    for k, v in overrides.items():
        if k not in train:
            raise ValueError(f"Unknown setting '{k}'. Choose from {list(train)}")
        train[k] = v
    return dict(train=train, gen=gen, n_stories=n_stories, max_stories=max_stories,
                out_dir=out_dir, smoke=smoke)