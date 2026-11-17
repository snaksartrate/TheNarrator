# Evaluation

Run from the repository root. Reads saved artifacts only; never trains.

    python -m evaluation stories          # word metrics + timing (from generation_meta.json)
    python -m evaluation perplexity       # FINAL test perplexity; only after training ends
    python -m evaluation human-sheet      # blind_stories.csv, blind_key.csv, template
    python -m evaluation human-summary --ratings experiments/analysis/results/human_ratings.csv
    python -m evaluation compare          # model_comparison.csv (always six rows)
    python -m evaluation plots
    python -m pytest tests/test_evaluation.py -v

## Definitions
- Words: lowercase, letters/digits with internal apostrophes; punctuation excluded.
- Distinct-n = unique n-grams / total n-grams (0.0 if fewer than n words).
- Repetition = repeated trigram occurrences / total trigrams (n=3).
- Lexical diversity = unique / total words (length-sensitive; read with word_count).
- Perplexity = exp(token-weighted mean next-token CE) on the test split, using best.pt
  and its saved tokenizer. Token space: compare AdamW vs SGD within a tokenizer only.
  Windows do not overlap, so later windows of a long story lose earlier context.
  Bits/character is omitted (exact character alignment not guaranteed).
- Timing is taken from generation_meta.json (no second generation pass); depends on CPU load.

## Human protocol
Six criteria, 1-5. Raters get blind_stories.csv only (shuffled, seed 42); blind_key.csv stays
with the organiser. One row per evaluator x story; raw ratings are kept. "Relevance" = fit to the
story's own established premise/theme (generation is unconditional, there is no external prompt).
Scoring anchors (1/3/5 per criterion) must be added to the rater instructions before scoring.

## Exclusions
Smoke runs (`smoke_test: true`) and anything outside the six IDs are rejected.
Outputs go to experiments/analysis/ (git-ignored); training folders are never written.