"Data loader for midtraining — streams batches from a TaskMixture."

from collections import deque
import torch

def mid_data_generator(dataset, tokenizer, batch_size, seq_len, device="cuda"):
    "Yields (inputs, targets) batches from a conversation dataset."
    device_type = "cuda" if "cuda" in device else "cpu"
    needed_tokens = batch_size * seq_len + 1
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=(device_type == "cuda"))
    cursor = 0
    dataset_size = len(dataset)

    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor = (cursor + 1) % dataset_size

        for i in range(needed_tokens): scratch[i] = token_buffer.popleft()
        inputs = scratch[:-1].view(batch_size, seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = scratch[1:].view(batch_size, seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets
