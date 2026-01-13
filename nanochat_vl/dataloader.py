"Streaming tokenizing data loader for pretraining."

from collections import deque
import torch, pyarrow.parquet as pq
from nanochat_vl.dataset import list_parquet_files
from nanochat_vl.tokenizer import get_tokenizer

def data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    "Stream parquet files, tokenize on-the-fly, yield (inputs, targets) batches."
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    assert parquet_paths, "No parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    tokenizer, bos = get_tokenizer(), get_tokenizer().get_bos_token_id()
    token_buffer, needed = deque(), B * T + 1

    def doc_batches():
        while True:
            for path in parquet_paths:
                pf = pq.ParquetFile(path)
                for rg_idx in range(pf.num_row_groups):
                    batch = pf.read_row_group(rg_idx).column('text').to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size): yield batch[i:i+tokenizer_batch_size]

    batches = doc_batches()
    while True:
        while len(token_buffer) < needed:
            for tokens in tokenizer.encode(next(batches), prepend=bos, num_threads=tokenizer_threads): token_buffer.extend(tokens)
        tokens = [token_buffer.popleft() for _ in range(needed)]
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=(device=="cuda"))
        inputs, targets = scratch[:-1].view(B, T), scratch[1:].view(B, T)
        yield inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
