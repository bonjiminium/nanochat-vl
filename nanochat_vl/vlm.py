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

class Projector(nn.Module):
    def __init__(self, vision_dim=256, lm_dim=128):
        super().__init__()
        self.proj = nn.Linear(vision_dim, lm_dim)
    
    def forward(self, x): return self.proj(x)

class VLM(nn.Module):
    def __init__(self, gpt, img_size=64, patch_size=8, vision_dim=256):
        super().__init__()
        self.gpt, self.img_size = gpt, img_size
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=vision_dim)
        self.proj = Projector(vision_dim, gpt.config.n_embd)
    
    def forward(self, img, input_ids, targets=None):
        img_emb = self.proj(self.patch_embed(img))
        tok_emb = norm(self.gpt.transformer.wte(input_ids))
        img_emb = img_emb.to(tok_emb.dtype)
        x = torch.cat([img_emb, tok_emb], dim=1)
        B, T, _ = x.size()
        cos_sin = self.gpt.cos[:, :T], self.gpt.sin[:, :T]
        for block in self.gpt.transformer.h: x = block(x, cos_sin)
        x = norm(x)
        logits = self.gpt.lm_head(x)[..., :self.gpt.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is None: return logits
        txt_logits = logits[:, img_emb.size(1):, :]
        return F.cross_entropy(txt_logits.reshape(-1, txt_logits.size(-1)), targets.reshape(-1), ignore_index=-1)
