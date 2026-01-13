"Engine for efficient inference."
import torch, torch.nn.functional as F

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    if temperature == 0.0: return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        probs = F.softmax(vals / temperature, dim=-1)
        return idx.gather(1, torch.multinomial(probs, num_samples=1, generator=rng))
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1, generator=rng)

class Engine:
    def __init__(self, model, tokenizer): self.model, self.tokenizer = model, tokenizer

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=64, temperature=0.0, top_k=None, seed=42):
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        bos = self.tokenizer.get_bos_token_id()
        tokens = self.tokenizer(prompt, prepend=bos) if isinstance(prompt, str) else prompt
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = self.model(ids)
            next_id = sample_next_token(logits[:, -1, :], rng, temperature, top_k)
            if next_id.item() == bos: break
            ids = torch.cat([ids, next_id], dim=1)
        return self.tokenizer.decode(ids[0, len(tokens):].tolist())
