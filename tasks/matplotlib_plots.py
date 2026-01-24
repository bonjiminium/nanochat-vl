"Matplotlib plots dataset for ViT pretraining."

from datasets import load_dataset
from tasks.common import Task

class MatplotlibPlots(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        split_map = dict(train="train", val="validation", validation="validation", test="test")
        self.ds = load_dataset("bxw315-umd/matplotlib-plots", split=split_map[split])

    @property
    def eval_type(self): return 'categorical'

    def num_examples(self): return len(self.ds)

    def get_example(self, idx):
        row = self.ds[idx]
        return dict(image=row["image"], color=row["color"], y_exponent=row["y_exponent"])
