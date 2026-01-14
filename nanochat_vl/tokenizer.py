"BPE Tokenizer in the style of GPT-4, using rustbpe for training and tiktoken for inference."

import os, copy, pickle
from functools import lru_cache
import rustbpe, tiktoken

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RustBPETokenizer:
    "Light wrapper around tiktoken (for efficient inference) but train with rustbpe"
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f: enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self): return self.enc.n_vocab
    def get_special_tokens(self): return self.enc.special_tokens_set
    def id_to_token(self, id): return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text): return self.enc.encode_single_token(text)

    def get_bos_token_id(self): return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None: prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None: append_id = append if isinstance(append, int) else self.encode_special(append)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None: ids.insert(0, prepend_id)
            if append is not None: ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids: ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids: ids_row.append(append_id)
        else: raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs): return self.encode(*args, **kwargs)
    def decode(self, ids): return self.enc.decode(ids)

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation.
        Returns:
        - ids: list[int] of token ids
        - mask: list[int] of same length, mask=1 for tokens the assistant should train on
        """
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int): token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge system message into first user message if present
        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        # Special tokens
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")

        # Render
        add_tokens(bos, 0)
        for message in messages:
            content = message["content"]
            if message["role"] == "user":
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                add_tokens(self.encode(content), 1)  # mask=1: supervise these
                add_tokens(assistant_end, 1)

        return ids[:max_tokens], mask[:max_tokens]

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f: pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

def get_tokenizer():
    from nanochat_vl.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat_vl.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f: token_bytes = torch.load(f, map_location=device)
    return token_bytes
