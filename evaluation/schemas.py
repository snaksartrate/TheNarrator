"""Shared names, paths and column lists. One source of truth for CSV/JSON/plots."""
import config

EXPERIMENT_IDS = list(config.EXPERIMENTS)          # fixed order: adamw_char ... sgd_bpe
CRITERIA = ["grammar", "coherence", "creativity", "continuity", "relevance", "completeness"]

ANALYSIS_DIR = config.ROOT / "experiments" / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
VIS_DIR = ANALYSIS_DIR / "visualizations"

STORY_COLUMNS = [
    "experiment_id", "optimizer", "tokenizer", "vocab_size", "checkpoint", "checkpoint_step",
    "story_id", "story_file", "generation_seed", "generation_settings", "dataset", "device",
    "word_count", "distinct_1", "distinct_2", "repetition_rate", "repetition_ngram_n",
    "lexical_diversity", "generation_time_seconds", "seconds_per_word", "words_per_second",
    "generated_token_count",
]
HUMAN_RAW_COLUMNS = ["evaluator_id", "blind_story_id", *CRITERIA, "comments"]