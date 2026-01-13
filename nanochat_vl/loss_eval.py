"Functions for evaluating a base model."
import math, torch

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    "Returns bits per byte (bpb) - a vocab-size-independent metric."
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    for _ in range(steps):
        x, y = next(batches)
        loss = model(x, y, loss_reduction='none')
        y = y.view(-1)
        num_bytes = token_bytes[y] if (y >= 0).all() else token_bytes[y.clamp(min=0)] * (y >= 0)
        total_nats += (loss * (num_bytes > 0)).sum()
        total_bytes += num_bytes.sum()
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    if total_bytes == 0: return float('inf')
    return (total_nats / total_bytes) / math.log(2)
