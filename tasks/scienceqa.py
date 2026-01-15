"ScienceQA multiple-choice science questions with images from derek-thomas/ScienceQA."

from datasets import load_dataset
from tasks.common import Task, render_mc

class ScienceQA(Task):
    def __init__(self, split="test", only_images=True, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "ScienceQA split must be train|validation|test"
        self.ds = load_dataset("derek-thomas/ScienceQA", split=split)
        if only_images: self.ds = self.ds.filter(lambda x: x["image"] is not None)

    @property
    def eval_type(self): return 'categorical'

    def num_examples(self): return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question, choices, answer = row["question"], row["choices"], row["answer"]
        letters = [chr(ord('A') + i) for i in range(len(choices))]
        user_message = render_mc(question, letters, choices)
        messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": letters[answer]}]
        return dict(messages=messages, letters=letters, image=row["image"])

    def evaluate(self, conversation, assistant_response):
        return assistant_response == conversation['messages'][-1]['content']
