# tokenizers

Tokenization module of TheNarrator. Three tokenizers behind one interface, so the
model and training code can swap them without changes.

## Interface (mandatory)

```python
tokenizer.train(text)             # build vocabulary from TRAIN text only
ids  = tokenizer.encode(text)     # list[int]
text = tokenizer.decode(ids)      # str
n    = tokenizer.vocab_size       # int
```

Extras that do not change the interface:

* `encode(text, add_special_tokens=True)` adds BOS at the start and EOS at the end
* `decode(ids, skip_special_tokens=True)`
* `tokenizer.pad_id / unk_id / bos_id / eos_id`
* `tokenizer.save(path)` and `load_tokenizer(path)`
* `get_tokenizer("char" | "word" | "bpe", **kwargs)`
* `tokenizer.get_stats(text)` returns vocab size, token count, chars per token, unk rate

Special tokens always have the same ids: `<pad>=0, <unk>=1, <bos>=2, <eos>=3`.

## Tokenizers

| File | Class | Vocabulary | Notes |
|------|-------|------------|-------|
| char_tokenizer.py | CharTokenizer | every distinct character | tiny vocab, long sequences |
| word_tokenizer.py | WordTokenizer | words and punctuation with freq >= min_freq | short sequences, large vocab, unk problem |
| bpe_tokenizer.py | BPETokenizer | characters plus learned merges up to vocab_size | middle ground, no unknown words |

Word decoding rebuilds spaces with simple rules, so it is not perfectly lossless.
Char and BPE decoding is exactly lossless for characters seen in training.

## Usage

```python
from tokenizers import get_tokenizer, load_tokenizer

tok = get_tokenizer("bpe", vocab_size=4000)
tok.train(open("dataset/processed/train.txt", encoding="utf-8").read())
tok.save("tokenizers/saved/bpe_tokenizer.json")

tok = load_tokenizer("tokenizers/saved/bpe_tokenizer.json")
ids = tok.encode("Once upon a time", add_special_tokens=True)
print(tok.decode(ids))
```

## Fit all three and print stats

Run from the repository root:

```
python -m tokenizers.train_tokenizers
python -m tokenizers.train_tokenizers --bpe-vocab 4000 --min-freq 3
```

Outputs go to `tokenizers/saved/` (git ignored): three vocab files plus
`tokenizer_stats.json` and `tokenizer_stats.csv`.

## Tests

```
python -m pytest tokenizers/tests -v
```

## Notes for the team

* Fit tokenizers on the training split only. Never on validation or test.
* Strip the story separator (for example <|endoftext|>) before train(), and encode each story with add_special_tokens=True so BOS and EOS mark story boundaries.
* Perplexity is per token, so it is not comparable across char, word and BPE.
  Use bits per character or compare within one tokenizer.
* Only the standard library is used (pytest for tests).
* The folder name `tokenizers` matches the HuggingFace package name. Always run
  from the repository root so the local folder wins on import.
