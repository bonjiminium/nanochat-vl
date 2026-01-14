"The ARC dataset from Allen AI. https://huggingface.co/datasets/allenai/ai2_arc"

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self): return 'categorical'

    def num_examples(self): return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question, choices, answer_string = row["question"], row["choices"]["text"], row["answerKey"]
        letters = row["choices"]["label"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}"
        user_message = render_mc(question, letters, choices)
        messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer_string}]
        return dict(messages=messages, letters=letters)

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        return assistant_response == conversation['messages'][-1]['content']
