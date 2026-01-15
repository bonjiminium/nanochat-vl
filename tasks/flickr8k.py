"Flickr8k image-caption dataset from jxie/flickr8k on HuggingFace."

from datasets import load_dataset
from pathlib import Path

class Flickr8k:
    def __init__(self, split="train", cache_dir=None):
        assert split in ["train", "test"], "Flickr8k split must be train|test"
        self.ds = load_dataset("jxie/flickr8k", split=split)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "flickr8k"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return row["caption_4"], row["image"]
