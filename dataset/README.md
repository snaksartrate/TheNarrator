# Dataset

## Dataset source

This dataset is sourced from [roneneldan/TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories). Only the first 30,000 stories of the `train` split are used.

## License

`cdla-sharing-1.0`

## Cleaning steps

Applied in this order:

1. Convert curly quotes and dashes to plain ASCII.
2. Strip accents.
3. Keep only newline and printable ASCII.
4. Collapse repeated spaces and blank lines.
5. Drop empty stories.
6. Drop stories under 100 characters.
7. Drop stories containing the separator.
8. Remove duplicates.

## Splitting

Stories are shuffled with seed 42 and split 80/10/10 into train, validation, and test sets by whole story. There is no overlap between splits.

## File format

Stories in `train.txt`, `validation.txt`, and `test.txt` are separated by a line containing `<|endofstory|>`.

## Statistics

| Split | Stories | Characters | Words | Average words/story | Average characters/story | Unique characters |
|---|---:|---:|---:|---:|---:|---:|
| Train | 24,000 | 21,325,769 | 4,176,126 | 174.0 | 888.6 | 80 |
| Validation | 3,000 | 2,662,157 | 521,742 | 173.9 | 887.4 | 70 |
| Test | 3,000 | 2,661,219 | 521,313 | 173.8 | 887.1 | 73 |

Additional recorded vocabulary counts: 11,971 unique words in train and 80 unique characters in train. The cleaning report records 30,000 raw stories and 30,000 final stories; it records zero removed empty stories, zero removed stories under the minimum length, zero removed stories containing the separator, and zero removed duplicates.

## Regenerate and verify

From the repository root, regenerate the dataset with:

```bash
python -m src.data.preprocess
```

Verify the generated files with:

```bash
python -m src.data.validate
```

## Use the loader

`load_split` returns a list of stories. Encode the training stories with your tokenizer, then pass the resulting token IDs to `StoryDataset`:

```python
from src.data import StoryDataset, load_split

stories = load_split("train")

token_ids = []

for story in stories:
    story_ids = tokenizer.encode(
        story,
        add_special_tokens=True
    )
    token_ids.extend(story_ids)

train_dataset = StoryDataset(
    token_ids,
    block_size=128
)
```

## Note for tokenizer authors

Train BPE and any vocabulary on `train.txt` only.
