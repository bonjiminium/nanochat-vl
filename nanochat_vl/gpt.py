"GPT model for nanochat-vl, ported from Karpathy's nanochat."

import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class GPTConfig:
    seq_len: int = 1024
    vocab_size: int = 65536
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

def norm(x): return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1, y2 = x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx, self.n_head, self.n_kv_head, self.n_embd = layer_idx, config.n_head, config.n_kv_head, config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None: k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        enable_gqa = self.n_head != self.n_kv_head
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x): return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn, self.mlp = CausalSelfAttention(config, layer_idx), MLP(config)

    def forward(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        return x + self.mlp(norm(x))

class GPT(nn.Module):
    def __init__(self, config, pad_vocab=64):
        super().__init__()
        self.config = config
        padded = ((config.vocab_size + pad_vocab - 1) // pad_vocab) * pad_vocab
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(padded, config.n_embd), h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)])))
        self.lm_head = nn.Linear(config.n_embd, padded, bias=False)
        self.rotary_seq_len = config.seq_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        cos, sin = self._precompute_rotary(self.rotary_seq_len, self.config.n_embd // self.config.n_head)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda": self.transformer.wte.to(dtype=torch.bfloat16)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = norm(self.transformer.wte(idx))
        for block in self.transformer.h: x = block(x, cos_sin, kv_cache)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None: return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

    def get_device(self): return self.transformer.wte.weight.device

    def num_params(self): return sum(p.numel() for p in self.parameters())

    def estimate_flops(self):
        nparams_emb = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.seq_len
        return 6 * (self.num_params() - nparams_emb) + 12 * l * h * q * t

    def setup_optimizers(self, embedding_lr=0.3, unembedding_lr=0.004, matrix_lr=0.02, adam_betas=(0.9, 0.95), weight_decay=0.0):
        from nanochat_vl.muon import Muon
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        matrix_params = [p for n, p in self.named_parameters() if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n]
        adamw = torch.optim.AdamW([{'params': [self.transformer.wte.weight], 'lr': embedding_lr * dmodel_lr_scale}, {'params': [self.lm_head.weight], 'lr': unembedding_lr * dmodel_lr_scale}], betas=adam_betas, weight_decay=weight_decay, fused=True, eps=1e-10)
        muon = Muon(matrix_params, lr=matrix_lr * dmodel_lr_scale, momentum=0.95)
        optimizers = [adamw, muon]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers
