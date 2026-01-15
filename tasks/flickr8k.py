"Flickr8k image-caption dataset from jxie/flickr8k on HuggingFace."

from datasets import load_dataset
from tasks.common import Task

class Flickr8k(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "Flickr8k split must be train|test"
        self.ds = load_dataset("jxie/flickr8k", split=split)

    def num_examples(self): return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = [{"role": "user", "content": "Describe this image."}, {"role": "assistant", "content": row["caption_4"]}]
        return {"messages": messages, "image": row["image"]}
