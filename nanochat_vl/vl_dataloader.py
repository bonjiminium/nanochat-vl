import torch, numpy as np
from PIL import Image

def process_image(img, size):
    if isinstance(img, str): img = Image.open(img)
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5

def vl_data_generator(dataset, tokenizer, batch_size, img_size, max_seq_len, device="cuda"):
    cursor = 0
    while True:
        batch_imgs, batch_ids, batch_targets = [], [], []
        for _ in range(batch_size):
            caption, img_data = dataset[cursor]
            img = torch.randn(3, img_size, img_size) if img_data is None else process_image(img_data, img_size)
            ids = tokenizer.encode(caption)
            batch_imgs.append(img)
            batch_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            cursor = (cursor + 1) % len(dataset)
        max_len = max_seq_len
        pad_id = tokenizer.encode_special("<|assistant_end|>")
        inputs = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
        for i, (ids, tgt) in enumerate(zip(batch_ids, batch_targets)):
            length = min(len(ids), max_len)
            inputs[i, :length] = torch.tensor(ids[:length])
            targets[i, :length] = torch.tensor(tgt[:length])
        imgs = torch.stack(batch_imgs)
        yield imgs.to(device), inputs.to(device), targets.to(device)
