"""One-off: re-generate stories for a finished run. Does not train."""
import sys
import config
from src.generator import GenConfig, generate_and_save

eid = sys.argv[1]
run = config.OUTPUT_DIR / eid
meta = generate_and_save(run, GenConfig(**config.GEN), n_stories=config.N_STORIES, device="cpu")
print([s["num_tokens"] for s in meta["stories"]])