"""Reusable loader for the processed story dataset.

This file does NOT import preprocess.py, so the model team can use it
without any of the collection / cleaning code.
"""
from .constants import PROCESSED_DIR, STORY_SEP

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # lets load_split work even if torch is missing
    torch = None
    Dataset = object


def load_split(name, processed_dir=PROCESSED_DIR):
    """Return the stories of 'train', 'validation' or 'test' as a list of strings."""
    if name not in ("train", "validation", "test"):
        raise ValueError("name must be 'train', 'validation' or 'test'")
    with open(f"{processed_dir}/{name}.txt", encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in text.split(STORY_SEP) if s.strip()]


class StoryDataset(Dataset):
    """Turns a flat list of token ids into (x, y) next-token training pairs.

    y is x shifted one position to the right.
    """

    def __init__(self, token_ids, block_size):
        if torch is None:
            raise ImportError("PyTorch is required for StoryDataset")
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max((len(self.ids) - 1) // self.block_size, 0)

    def __getitem__(self, i):
        start = i * self.block_size
        x = self.ids[start:start + self.block_size]
        y = self.ids[start + 1:start + self.block_size + 1]
        return x, y
