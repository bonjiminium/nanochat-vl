"GSM8K task for training and evaluation."

import re
from datasets import load_dataset
from tasks.common import Task

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return match_str
    return None

class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self): return 'generative'
    def num_examples(self): return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question, answer = row['question'], row['answer']
        assistant_message_parts = []
        for part in re.split(r'(<<[^>]+>>)', answer):
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                expr, result = inner.rsplit('=', 1) if '=' in inner else (inner, "")
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else: assistant_message_parts.append({"type": "text", "text": part})
        return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": assistant_message_parts}]}

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str)
        last_text_part = conversation['messages'][-1]['content'][-1]['text']
        return int(extract_answer(assistant_response) == extract_answer(last_text_part))

    def reward(self, conversation, assistant_response): return float(self.evaluate(conversation, assistant_response))
