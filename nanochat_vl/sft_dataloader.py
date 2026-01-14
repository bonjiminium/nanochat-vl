"Data loader for SFT — only supervises assistant tokens."

import torch

def sft_data_generator(dataset, tokenizer, batch_size, max_seq_len, device="cuda"):
    "Yields (inputs, targets) batches where targets=-1 for non-assistant tokens."
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    cursor = 0
    dataset_size = len(dataset)

    while True:
        batch = []
        while len(batch) < batch_size:
            doc = dataset[cursor]
            ids, mask = tokenizer.render_conversation(doc, max_tokens=max_seq_len + 1)
            if len(ids) > 1: batch.append((ids, mask))
            cursor = (cursor + 1) % dataset_size

        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((batch_size, ncols), pad_token_id, dtype=torch.int32)
        targets = torch.full((batch_size, ncols), -1, dtype=torch.int64)

        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.int32)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = torch.tensor(ids[1:], dtype=torch.int64)
            mask_tensor = torch.tensor(mask[1:], dtype=torch.int64)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets

        yield inputs.to(device=device, non_blocking=True), targets.to(device=device, non_blocking=True)
