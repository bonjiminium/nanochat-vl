import torch, numpy as np
from PIL import Image

IMAGE_PLACEHOLDER = "<|image|>"

def process_image(img, size):
    if isinstance(img, str): img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (128, 128, 128))
    padded.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    x = torch.tensor(np.array(padded)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5

def tokenize_with_images(text, tokenizer, img_token_id, num_patches):
    parts = text.split(IMAGE_PLACEHOLDER)
    ids, img_positions = [], []
    for i, part in enumerate(parts):
        ids.extend(tokenizer.encode(part))
        if i < len(parts) - 1:
            img_positions.append(len(ids))
            ids.extend([img_token_id] * num_patches)
    return ids, img_positions

def vl_data_generator(dataset, tokenizer, batch_size, img_size, max_seq_len, num_patches, device="cuda"):
    cursor = 0
    img_token_id = tokenizer.encode_special("<image>")
    pad_id = tokenizer.encode_special("<|assistant_end|>")
    while True:
        batch_imgs, batch_ids, batch_targets = [], [], []
        for _ in range(batch_size):
            example = dataset[cursor]
            ids, _ = tokenizer.render_conversation_vl(example, img_token_id, num_patches)
            images = example.get("images", [])
            num_images = ids.count(img_token_id) // num_patches
            for i in range(num_images):
                img_data = images[i] if i < len(images) else None
                img = torch.randn(3, img_size, img_size) if img_data is None else process_image(img_data, img_size)
                batch_imgs.append(img)
            batch_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            cursor = (cursor + 1) % len(dataset)
        inputs = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        targets = torch.full((batch_size, max_seq_len), -1, dtype=torch.long)
        for i, (ids, tgt) in enumerate(zip(batch_ids, batch_targets)):
            length = min(len(ids), max_seq_len)
            inputs[i, :length] = torch.tensor(ids[:length])
            targets[i, :length] = torch.tensor(tgt[:length])
        imgs = torch.stack(batch_imgs) if batch_imgs else torch.empty(0, 3, img_size, img_size)
        yield imgs.to(device), inputs.to(device), targets.to(device)
