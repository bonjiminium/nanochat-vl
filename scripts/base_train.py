"Minimal single-GPU base training script."

import os, time, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.dataloader import data_loader
from nanochat_vl.muon import Muon
from nanochat_vl.common import get_base_dir
from nanochat_vl.tokenizer import get_token_bytes
from nanochat_vl.loss_eval import evaluate_bpb

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--n_head', type=int, default=6)
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--vocab_size', type=int, default=65536)
parser.add_argument('--device_batch_size', type=int, default=16)
parser.add_argument('--total_batch_size', type=int, default=65536)
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--embedding_lr', type=float, default=0.3)
parser.add_argument('--unembedding_lr', type=float, default=0.004)
parser.add_argument('--matrix_lr', type=float, default=0.02)
parser.add_argument('--warmup_iters', type=int, default=100)
parser.add_argument('--cooldown_iters', type=int, default=400)
parser.add_argument('--eval_every', type=int, default=250)
parser.add_argument('--eval_tokens', type=int, default=20*524288)
args = parser.parse_args()

assert args.total_batch_size % args.device_batch_size == 0
grad_accum_steps = args.total_batch_size // args.device_batch_size

cfg = GPTConfig(seq_len=args.max_seq_len, vocab_size=args.vocab_size, n_layer=args.depth, n_head=args.n_head, n_kv_head=args.n_head, n_embd=args.n_embd)
model = GPT(cfg).cuda().bfloat16()
model.init_weights()
print(f"Model: {model.num_params():,} parameters")

matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n]
adamw = torch.optim.AdamW([{'params': [model.transformer.wte.weight], 'lr': args.embedding_lr}, {'params': [model.lm_head.weight], 'lr': args.unembedding_lr}], betas=(0.9, 0.95), weight_decay=0.0)
muon = Muon(matrix_params, lr=args.matrix_lr, momentum=0.95)

def get_lr_mult(step):
    if step < args.warmup_iters: return step / args.warmup_iters
    if step >= args.num_iterations - args.cooldown_iters: return (args.num_iterations - step) / args.cooldown_iters
    return 1.0

train_loader = data_loader(args.device_batch_size, args.max_seq_len, 'train', device='cuda')
model.train()

token_bytes = get_token_bytes(device='cuda')

def run_eval(step):
    model.eval()
    val_loader = data_loader(args.device_batch_size, args.max_seq_len, 'val', device='cuda')
    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
    bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print(f"step {step:4d} | val_bpb {bpb:.4f}")
    model.train()

for step in range(args.num_iterations):
    if step % args.eval_every == 0: run_eval(step)
    t0 = time.time()
    lr_mult = get_lr_mult(step)
    adamw.param_groups[0]['lr'] = args.embedding_lr * lr_mult
    adamw.param_groups[1]['lr'] = args.unembedding_lr * lr_mult
    for g in muon.param_groups: g['lr'] = args.matrix_lr * lr_mult
    
    total_loss = 0.0
    for _ in range(grad_accum_steps):
        x, y = next(train_loader)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()
        total_loss += loss.item()
    
    muon.step()
    adamw.step()
    muon.zero_grad()
    adamw.zero_grad()
    
    dt = time.time() - t0
    tokens_per_sec = args.total_batch_size * args.max_seq_len / dt
    print(f"step {step:4d} | loss {total_loss:.4f} | lr_mult {lr_mult:.2f} | {tokens_per_sec:.0f} tok/s | {dt*1000:.0f}ms")
