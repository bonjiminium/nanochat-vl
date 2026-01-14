"The MMLU dataset. https://huggingface.co/datasets/cais/mmlu"

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train": assert split == "train", "auxiliary_train must be split into train"
        self.subset, self.split = subset, split
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train": self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

    @property
    def eval_type(self): return 'categorical'
    def num_examples(self): return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question, choices, answer = row["question"], row["choices"], row["answer"]
        user_message = render_mc(question, self.letters, choices)
        return dict(messages=[{"role": "user", "content": user_message}, {"role": "assistant", "content": self.letters[answer]}], letters=self.letters)

    def evaluate(self, conversation, assistant_response):
        expected = conversation["messages"][-1]["content"]
        return assistant_response.strip() == expected
