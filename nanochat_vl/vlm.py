import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(x): return F.rms_norm(x, (x.size(-1),))

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp_up = nn.Linear(dim, 4 * dim, bias=False)
        self.mlp_down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(norm(x)).reshape(B, N, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = x + F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C) @ self.proj.weight.T
        x = x + F.gelu(self.mlp_up(norm(x))) @ self.mlp_down.weight.T
        return x

class ViT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, dim=256, n_layer=4, n_head=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, n_head) for _ in range(n_layer)])

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return norm(x)

class Projector(nn.Module):
    def __init__(self, vision_dim=256, lm_dim=128):
        super().__init__()
        self.proj = nn.Linear(vision_dim, lm_dim)
    
    def forward(self, x): return self.proj(x)

class VLM(nn.Module):
    def __init__(self, gpt, img_size=64, patch_size=8, vision_dim=256, vit_layers=4, vit_heads=4):
        super().__init__()
        self.gpt, self.img_size = gpt, img_size
        self.vit = ViT(img_size, patch_size, 3, vision_dim, vit_layers, vit_heads)
        self.proj = Projector(vision_dim, gpt.config.n_embd)
    
    def forward(self, imgs, input_ids, targets=None, img_token_id=None):
        B, T = input_ids.shape
        tok_emb = norm(self.gpt.transformer.wte(input_ids))
        if imgs is not None and imgs.numel() > 0 and img_token_id is not None:
            img_emb = self.proj(self.vit(imgs.to(tok_emb.dtype))).to(tok_emb.dtype)
            img_emb = img_emb.view(-1, img_emb.size(-1))
            mask = (input_ids == img_token_id)
            num_img_tokens, num_img_emb = mask.sum().item(), img_emb.size(0)
            if num_img_tokens != num_img_emb: print(f"WARNING: image token mismatch! {num_img_tokens} tokens in input but {num_img_emb} image embeddings. Sequence truncation is dropping images - increase max_seq_len!")
            tok_emb[mask] = img_emb[:num_img_tokens]
        x = tok_emb
        cos_sin = self.gpt.cos[:, :T], self.gpt.sin[:, :T]
        for block in self.gpt.transformer.h: x = block(x, cos_sin)
        x = norm(x)
        logits = self.gpt.lm_head(x)[..., :self.gpt.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is None: return logits
        if img_token_id is not None: targets = targets.masked_fill(targets == img_token_id, -1)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
