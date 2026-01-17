"Muon optimizer from Keller et al."

import torch
from torch import Tensor

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    "Newton-Schulz iteration to compute zeroth power / orthogonalization of G."
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        params = list(params)
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        param_groups = [dict(params=[p for p in params if p.numel() == size]) for size in {p.numel() for p in params}]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, nesterov, ns_steps = group['lr'], group['momentum'], group['nesterov'], group['ns_steps']
            for p in group['params']:
                g = p.grad
                if g is None: continue
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, ns_steps)
                p.add_(g, alpha=-lr * max(1, p.size(-2) / p.size(-1)) ** 0.5)
