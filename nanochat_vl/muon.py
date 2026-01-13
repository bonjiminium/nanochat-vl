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
    for _ in range(steps): X = a * X + b * (X @ X.mT) @ X + c * (X @ X.mT) @ (X @ X.mT) @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

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
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, ns_steps)
                g *= max(1, g.size(-2) / g.size(-1)) ** 0.5
                p.add_(g, alpha=-lr)
