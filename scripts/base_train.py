"Minimal single-GPU base training script."

import os, time, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.dataloader import data_loader

from nanochat_vl.common import get_base_dir, DummyWandb
from nanochat_vl.tokenizer import get_token_bytes
from nanochat_vl.loss_eval import evaluate_bpb
from nanochat_vl.tokenizer import get_tokenizer
from scripts.base_eval import evaluate_model
from nanochat_vl.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--aspect_ratio', type=int, default=64)
parser.add_argument('--head_dim', type=int, default=128)
parser.add_argument('--max_seq_len', type=int, default=2048)
parser.add_argument('--vocab_size', type=int, default=65536)
parser.add_argument('--device_batch_size', type=int, default=32)
parser.add_argument('--total_batch_size', type=int, default=524288)
parser.add_argument('--num_iterations', type=int, default=-1)
parser.add_argument('--target_param_data_ratio', type=int, default=8)
parser.add_argument('--embedding_lr', type=float, default=0.3)
parser.add_argument('--unembedding_lr', type=float, default=0.004)
parser.add_argument('--matrix_lr', type=float, default=0.02)
parser.add_argument('--warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmdown_ratio', type=float, default=0.4)
parser.add_argument('--final_lr_frac', type=float, default=0.0)
parser.add_argument('--eval_every', type=int, default=250)
parser.add_argument('--eval_tokens', type=int, default=20*524288)
parser.add_argument('--core_metric_every', type=int, default=2000)
parser.add_argument('--core_max_per_task', type=int, default=500)
parser.add_argument('--run', type=str, default='dummy')
parser.add_argument('--save_every', type=int, default=-1)
parser.add_argument('--resume_from', type=int, default=-1)
args = parser.parse_args()

def find_num_heads(model_dim, target_head_dim):
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0: return candidate
    return 1

model_dim = args.depth * args.aspect_ratio
num_heads = find_num_heads(model_dim, args.head_dim)

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
assert args.total_batch_size % tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be divisible by device_batch_size * max_seq_len ({tokens_per_fwdbwd})"
grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", f"d{args.depth}")
start_step = 0

cfg = GPTConfig(seq_len=args.max_seq_len, vocab_size=args.vocab_size, n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim)
model = GPT(cfg).cuda().bfloat16()
model.init_weights()
model = torch.compile(model)
num_params = model.num_params()
num_scaling_params = model.num_scaling_params()
print(f"Model: {num_params:,} parameters (scaling: {num_scaling_params:,})")
num_flops_per_token = model.estimate_flops()

assert args.num_iterations > 0 or args.target_param_data_ratio > 0
if args.num_iterations > 0: num_iterations = args.num_iterations
elif args.target_param_data_ratio > 0:
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // args.total_batch_size
    print(f"Calculated num_iterations from target_param_data_ratio: {num_iterations:,}")
get_max_memory = torch.cuda.max_memory_allocated
from nanochat_vl.report import get_gpu_info, get_gpu_tflops
promised_flops_per_sec = get_gpu_tflops(get_gpu_info())

reference_batch_size = 2**19
batch_lr_scale = (args.total_batch_size / reference_batch_size) ** 0.5
adamw, muon = model.setup_optimizers(embedding_lr=args.embedding_lr * batch_lr_scale, unembedding_lr=args.unembedding_lr * batch_lr_scale, matrix_lr=args.matrix_lr * batch_lr_scale)

if args.resume_from >= 0 or (args.resume_from == -1 and os.path.exists(checkpoint_dir)):
    resume_step = args.resume_from if args.resume_from >= 0 else find_last_step(checkpoint_dir)
    if resume_step is not None:
        model_data, optim_data, meta_data = load_checkpoint(checkpoint_dir, resume_step, 'cuda', load_optimizer=True)
        model.load_state_dict(model_data)
        adamw.load_state_dict(optim_data['adamw'])
        muon.load_state_dict(optim_data['muon'])
        start_step = resume_step + 1
        print(f"Resumed from step {resume_step}")

def get_lr_mult(step):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if step < warmup_iters: return (step + 1) / warmup_iters
    if step <= num_iterations - warmdown_iters: return 1.0
    progress = (num_iterations - step) / warmdown_iters
    return progress * 1.0 + (1 - progress) * args.final_lr_frac

train_loader = data_loader(args.device_batch_size, args.max_seq_len, 'train', device='cuda')
model.train()

token_bytes = get_token_bytes(device='cuda')

tokenizer = get_tokenizer()

if args.run == 'dummy': wandb_run = DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project='nanochat-vl', name=args.run, config=vars(args))

val_bpb, min_val_bpb = None, float('inf')

def run_eval(step):
    global val_bpb, min_val_bpb
    model.eval()
    val_loader = data_loader(args.device_batch_size, args.max_seq_len, 'val', device='cuda')
    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    if val_bpb < min_val_bpb: min_val_bpb = val_bpb
    print(f"step {step:4d} | val_bpb {val_bpb:.4f}")
    wandb_run.log(dict(step=step, val_bpb=val_bpb))
    model.train()

def run_core_eval(step):
    model.eval()
    results = evaluate_model(model, tokenizer, 'cuda', max_per_task=args.core_max_per_task)
    print(f"step {step:4d} | core_metric {results['core_metric']:.4f}")
    wandb_run.log(dict(step=step, core_metric=results['core_metric']))
    model.train()

def do_checkpoint(step):
    model_data = model.state_dict()
    optim_data = dict(adamw=adamw.state_dict(), muon=muon.state_dict())
    model_config = dict(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, n_layer=cfg.n_layer, n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd)
    meta_data = dict(step=step, val_bpb=val_bpb, min_val_bpb=min_val_bpb, config=vars(args), model_config=model_config)
    save_checkpoint(checkpoint_dir, step, model_data, optim_data, meta_data)
    print(f"Saved checkpoint at step {step}")

total_training_time, flops_so_far, mfu = 0.0, 0, 0.0
for step in range(start_step, num_iterations):
    if args.eval_every > 0 and step % args.eval_every == 0: run_eval(step)
    if args.core_metric_every > 0 and step % args.core_metric_every == 0: run_core_eval(step)
    t0 = time.time()
    lr_mult = get_lr_mult(step)
    for g in adamw.param_groups: g['lr'] = g['initial_lr'] * lr_mult
    for g in muon.param_groups: g['lr'] = g['initial_lr'] * lr_mult
    
    total_loss = 0.0
    for _ in range(grad_accum_steps):
        x, y = next(train_loader)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()
        total_loss += loss.item()
    
    muon_momentum = (1 - min(step/300, 1)) * 0.85 + min(step/300, 1) * 0.95
    for group in muon.param_groups: group["momentum"] = muon_momentum
    muon.step()
    adamw.step()
    torch.cuda.synchronize()
    muon.zero_grad()
    adamw.zero_grad()
    
    dt = time.time() - t0
    tokens_per_sec = args.total_batch_size / dt
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / promised_flops_per_sec
    if step > 10: total_training_time += dt
    flops_so_far = num_flops_per_token * args.total_batch_size * (step + 1)
    pct_done = 100 * step / (num_iterations - 1) if num_iterations > 1 else 100
    wandb_run.log(dict(step=step, loss=total_loss, lr_mult=lr_mult, tokens_per_sec=tokens_per_sec, mfu=mfu))
    print(f"step {step:5d}/{num_iterations-1} ({pct_done:5.2f}%) | loss {total_loss:.4f} | lr_mult {lr_mult:.2f} | dt {dt*1000:.0f}ms | tok/s {tokens_per_sec:,.0f} | mfu {mfu:.2f} | time {total_training_time/60:.2f}m")
    if args.save_every > 0 and (step + 1) % args.save_every == 0: do_checkpoint(step)

if args.save_every <= 0 or num_iterations % args.save_every != 0: do_checkpoint(num_iterations - 1)
wandb_run.finish()

from nanochat_vl.report import get_report
get_report().log(section="Base model training", data=[
    vars(args),
    dict(num_params=num_params, num_iterations=num_iterations, final_loss=total_loss, final_val_bpb=val_bpb, min_val_bpb=min_val_bpb,
         mfu_pct=f"{mfu:.2f}%", total_training_flops=f"{flops_so_far:.2e}", total_training_time=f"{total_training_time/60:.2f}m",
         peak_memory_mib=f"{get_max_memory() / 1024 / 1024:.2f}"),
])
