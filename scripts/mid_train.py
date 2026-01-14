"Minimal single-GPU midtraining script."

import os, time, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.muon import Muon
from nanochat_vl.common import get_base_dir, DummyWandb
from nanochat_vl.tokenizer import get_tokenizer, get_token_bytes
from nanochat_vl.loss_eval import evaluate_bpb
from nanochat_vl.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step, load_model
from nanochat_vl.mid_dataloader import mid_data_generator
from tasks.smoltalk import SmolTalk

parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_len', type=int, default=2048)
parser.add_argument('--device_batch_size', type=int, default=4)
parser.add_argument('--total_batch_size', type=int, default=32)
parser.add_argument('--num_iterations', type=int, default=100)
parser.add_argument('--embedding_lr', type=float, default=0.02)
parser.add_argument('--unembedding_lr', type=float, default=0.0004)
parser.add_argument('--matrix_lr', type=float, default=0.002)
parser.add_argument('--cooldown_iters', type=int, default=20)
parser.add_argument('--eval_every', type=int, default=25)
parser.add_argument('--eval_tokens', type=int, default=10*524288)
parser.add_argument('--run', type=str, default='dummy')
parser.add_argument('--save_every', type=int, default=-1)
parser.add_argument('--base_step', type=int, default=-1)
args = parser.parse_args()

assert args.total_batch_size % args.device_batch_size == 0
grad_accum_steps = args.total_batch_size // args.device_batch_size

base_dir = get_base_dir()

model, _, meta = load_model("base", "cuda", phase="train", step=args.base_step if args.base_step >= 0 else None)
model_config = GPTConfig(**meta["model_config"])
mid_checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", f"d{model_config.n_layer}")
model = model.bfloat16()
print(f"Loaded model: {model.num_params():,} parameters")

matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n]
adamw = torch.optim.AdamW([{'params': [model.transformer.wte.weight], 'lr': args.embedding_lr}, {'params': [model.lm_head.weight], 'lr': args.unembedding_lr}], betas=(0.9, 0.95), weight_decay=0.0)
muon = Muon(matrix_params, lr=args.matrix_lr, momentum=0.95)
for opt in [adamw, muon]:
    for g in opt.param_groups: g['initial_lr'] = g['lr']

def get_lr_mult(step):
    if step >= args.num_iterations - args.cooldown_iters: return (args.num_iterations - step) / args.cooldown_iters
    return 1.0

tokenizer = get_tokenizer()
train_ds = SmolTalk(split="train")
val_ds = SmolTalk(split="test", stop=1000)
train_loader = mid_data_generator(train_ds, tokenizer, args.device_batch_size, args.max_seq_len, device='cuda')
val_loader = mid_data_generator(val_ds, tokenizer, args.device_batch_size, args.max_seq_len, device='cuda')

token_bytes = get_token_bytes(device='cuda')
model.train()

if args.run == 'dummy': wandb_run = DummyWandb()
else:
    import wandb
    wandb_run = wandb.init(project='nanochat-vl', name=args.run, config=vars(args))

min_val_bpb = float('inf')
t0 = time.time()

for step in range(args.num_iterations):
    lr_mult = get_lr_mult(step)
    for opt in [adamw, muon]:
        for g in opt.param_groups: g['lr'] = g['initial_lr'] * lr_mult if 'initial_lr' in g else g['lr']

    for micro_step in range(grad_accum_steps):
        x, y = next(train_loader)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()

    adamw.step(); muon.step()
    adamw.zero_grad(); muon.zero_grad()

    if step % 1 == 0:
        dt = time.time() - t0
        print(f"step {step:4d} | loss {loss.item()*grad_accum_steps:.4f} | dt {dt*1000:.0f}ms")
        wandb_run.log(dict(step=step, loss=loss.item()*grad_accum_steps, lr_mult=lr_mult))
        t0 = time.time()

    if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, val_bpb)
        print(f"step {step:4d} | val_bpb {val_bpb:.4f} | min_val_bpb {min_val_bpb:.4f}")
        wandb_run.log(dict(step=step, val_bpb=val_bpb, min_val_bpb=min_val_bpb))
        model.train()

    if args.save_every > 0 and (step + 1) % args.save_every == 0:
        save_checkpoint(mid_checkpoint_dir, step, model.state_dict(), dict(adamw=adamw.state_dict(), muon=muon.state_dict()), dict(step=step, val_bpb=val_bpb, min_val_bpb=min_val_bpb, config=vars(args), model_config=model_config.__dict__))

final_loss = loss.item() * grad_accum_steps
print(f"Training complete. Final loss: {final_loss:.4f}, min_val_bpb: {min_val_bpb:.4f}")

from nanochat_vl.report import get_report
get_report().log(section="Midtraining", data=[vars(args), dict(num_iterations=args.num_iterations, min_val_bpb=min_val_bpb)])
