"A-OKVQA visual question answering from HuggingFaceM4/A-OKVQA."

from datasets import load_dataset
from tasks.common import Task, render_mc

class AOKVQA(Task):
    def __init__(self, split="train", use_images=True, **kwargs):
        super().__init__(**kwargs)
        split_map = dict(train="train", val="validation", validation="validation", test="test")
        self.ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split_map[split])
        self.use_images = use_images
        self.letters = "ABCDEFGHIJ"

    @property
    def eval_type(self): return 'categorical'

    def num_examples(self): return len(self.ds)

    def get_example(self, idx):
        row = self.ds[idx]
        choices = {c: self.letters[i] for i, c in enumerate(row["choices"])}
        answer = self.letters[row["correct_choice_idx"]]
        img_prefix = "<|image|>\n" if self.use_images else ""
        user_msg = f"{img_prefix}{render_mc(row['question'], choices)}"
        messages = [dict(role="user", content=user_msg), dict(role="assistant", content=answer)]
        images = [row["image"]] if self.use_images else []
        return dict(messages=messages, images=images)

    def get_example(self, index):
        row = self.ds[index]
        question, choices, answer_idx = row["question"], row["choices"], row["correct_choice_idx"]
        letters = [chr(ord('A') + i) for i in range(len(choices))]
        prefix = "<|image|>\n" if self.use_images else ""
        user_message = f"{prefix}{render_mc(question, letters, choices)}"
        messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": letters[answer_idx]}]
        images = [row["image"]] if self.use_images else []
        return dict(messages=messages, letters=letters, images=images)

    def evaluate(self, conversation, assistant_response): return assistant_response == conversation['messages'][-1]['content']
