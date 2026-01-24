"ViT pretraining on matplotlib plots with learnable class embeddings."

import os, argparse, time
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from nanochat_vl.vlm import ViT
from tasks.matplotlib_plots import MatplotlibPlots
from nanochat_vl.common import get_base_dir
from PIL import Image
import numpy as np

EXP_TO_IDX = {"1e−8": 0, "1e−7": 1, "1e−6": 2, "1e−5": 3, "1e−4": 4, "1e−3": 5, "1e−2": 6}

class ViTCLIP(nn.Module):
    def __init__(self, img_size=64, patch_size=8, dim=256, n_layer=4, n_head=4, n_colors=10, n_exponents=7):
        super().__init__()
        self.vit = ViT(img_size, patch_size, 3, dim, n_layer, n_head)
        self.color_embs = nn.Embedding(n_colors, dim)
        self.exp_embs = nn.Embedding(n_exponents, dim)
    
    def encode_image(self, x):
        x = self.vit(x)
        return F.normalize(x.mean(dim=1), dim=-1)
    
    def forward(self, x, color_labels, exp_labels):
        img_emb = self.encode_image(x)
        color_emb = F.normalize(self.color_embs.weight, dim=-1)
        exp_emb = F.normalize(self.exp_embs.weight, dim=-1)
        color_logits = img_emb @ color_emb.T
        exp_logits = img_emb @ exp_emb.T
        loss_color = F.cross_entropy(color_logits, color_labels)
        loss_exp = F.cross_entropy(exp_logits, exp_labels)
        return loss_color + loss_exp, color_logits, exp_logits

def process_image(img, size):
    if isinstance(img, Image.Image): img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (128, 128, 128))
    padded.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    x = torch.tensor(np.array(padded)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5

def collate_fn(batch, img_size):
    imgs = torch.stack([process_image(ex["image"], img_size) for ex in batch])
    colors = torch.tensor([ex["color"] for ex in batch])
    exps = torch.tensor([EXP_TO_IDX.get(ex["y_exponent"], 0) for ex in batch])
    return imgs, colors, exps

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_task = MatplotlibPlots(split="train")
    val_task = MatplotlibPlots(split="validation")
    train_loader = DataLoader(train_task.ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, args.img_size), num_workers=4)
    val_loader = DataLoader(val_task.ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, args.img_size), num_workers=4)
    
    model = ViTCLIP(img_size=args.img_size, patch_size=args.patch_size, dim=args.dim, n_layer=args.n_layer, n_head=args.n_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_color_correct, total_exp_correct, total = 0, 0, 0, 0
        t0 = time.time()
        for i, (imgs, colors, exps) in enumerate(train_loader):
            imgs, colors, exps = imgs.to(device), colors.to(device), exps.to(device)
            loss, color_logits, exp_logits = model(imgs, colors, exps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_color_correct += (color_logits.argmax(1) == colors).sum().item()
            total_exp_correct += (exp_logits.argmax(1) == exps).sum().item()
            total += imgs.size(0)
            if (i + 1) % args.log_every == 0:
                print(f"Epoch {epoch+1} Step {i+1}: loss={total_loss/total:.4f} color_acc={total_color_correct/total:.3f} exp_acc={total_exp_correct/total:.3f}")
        train_time = time.time() - t0
        print(f"Epoch {epoch+1} Train: loss={total_loss/total:.4f} color_acc={total_color_correct/total:.3f} exp_acc={total_exp_correct/total:.3f} time={train_time:.1f}s")
        
        model.eval()
        val_loss, val_color_correct, val_exp_correct, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, colors, exps in val_loader:
                imgs, colors, exps = imgs.to(device), colors.to(device), exps.to(device)
                loss, color_logits, exp_logits = model(imgs, colors, exps)
                val_loss += loss.item() * imgs.size(0)
                val_color_correct += (color_logits.argmax(1) == colors).sum().item()
                val_exp_correct += (exp_logits.argmax(1) == exps).sum().item()
                val_total += imgs.size(0)
        print(f"Epoch {epoch+1} Val: loss={val_loss/val_total:.4f} color_acc={val_color_correct/val_total:.3f} exp_acc={val_exp_correct/val_total:.3f}")
    
    if args.save_path:
        torch.save(model.vit.state_dict(), args.save_path)
        print(f"Saved ViT weights to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()
    main(args)
