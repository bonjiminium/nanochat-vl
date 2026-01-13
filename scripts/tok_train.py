"Train a tokenizer using rustbpe in the style of GPT-4."

import os, time, argparse, torch
from nanochat_vl.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat_vl.common import get_base_dir
from nanochat_vl.dataset import parquets_iter_batched
from nanochat_vl.report import get_report

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on')
parser.add_argument('--doc_cap', type=int, default=50_000, help='Maximum characters per document')
parser.add_argument('--vocab_size', type=int, default=32768, help='Vocabulary size')
args = parser.parse_args()

def text_iterator():
    total_chars = 0
    for batch in parquets_iter_batched("train"):
        for text in batch:
            text = text[:args.doc_cap]
            total_chars += len(text)
            yield text
            if total_chars >= args.max_chars: return

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator=text_iterator(), vocab_size=args.vocab_size)
t1 = time.time()
print(f"Training took {t1-t0:.2f} seconds")

base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(tokenizer_dir)
print(f"Saved tokenizer to {tokenizer_dir}")

special_set = set(SPECIAL_TOKENS)
token_bytes = torch.tensor([0 if tokenizer.decode([i]) in special_set else len(tokenizer.decode([i]).encode('utf-8')) for i in range(tokenizer.get_vocab_size())], dtype=torch.int32)
torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))
print(f"Saved token_bytes to {tokenizer_dir}/token_bytes.pt")

tb = token_bytes.float()
tb_nonzero = tb[tb > 0]
get_report().log(section="Tokenizer training", data=[
    vars(args),
    {"train_time": t1-t0},
    {"num_special_tokens": len(SPECIAL_TOKENS)},
    {"token_bytes_min": int(tb_nonzero.min().item()), "token_bytes_max": int(tb_nonzero.max().item()),
     "token_bytes_mean": tb_nonzero.mean().item(), "token_bytes_std": tb_nonzero.std().item()},
])
