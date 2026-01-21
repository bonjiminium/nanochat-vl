"Minimal single-GPU midtraining script."

import os, time, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.common import get_base_dir, DummyWandb
from nanochat_vl.tokenizer import get_tokenizer, get_token_bytes
from nanochat_vl.loss_eval import evaluate_bpb
from nanochat_vl.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step, load_model
from nanochat_vl.mid_dataloader import mid_data_generator
from tasks.smoltalk import SmolTalk

parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_len', type=int, default=2048)
parser.add_argument('--device_batch_size', type=int, default=4)
parser.add_argument('--total_batch_size', type=int, default=524288)
parser.add_argument('--num_iterations', type=int, default=100)
parser.add_argument('--embedding_lr', type=float, default=0.2)
parser.add_argument('--unembedding_lr', type=float, default=0.004)
parser.add_argument('--matrix_lr', type=float, default=0.02)

parser.add_argument('--eval_every', type=int, default=25)
parser.add_argument('--eval_tokens', type=int, default=20*524288)
parser.add_argument('--run', type=str, default='dummy')
parser.add_argument('--save_every', type=int, default=-1)
parser.add_argument('--base_step', type=int, default=-1)
args = parser.parse_args()

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
assert args.total_batch_size % tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be divisible by tokens_per_fwdbwd ({tokens_per_fwdbwd})"
grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

base_dir = get_base_dir()

model, _, meta = load_model("base", "cuda", phase="train", step=args.base_step if args.base_step >= 0 else None)
model_config = GPTConfig(**meta["model_config"])
mid_checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", f"d{model_config.n_layer}")
model = model.bfloat16()
model = torch.compile(model, dynamic=False)
print(f"Loaded model: {model.num_params():,} parameters")

adamw, muon = model.setup_optimizers(embedding_lr=args.embedding_lr, unembedding_lr=args.unembedding_lr, matrix_lr=args.matrix_lr)

def get_lr_mult(progress):
    return 1.0 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

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

min_val_bpb, val_bpb = float('inf'), float('inf')
num_flops_per_token = model.estimate_flops()
smooth_loss, total_time, ema_beta = 0.0, 0.0, 0.9
t0 = time.time()

for step in range(args.num_iterations):
    lr_mult = get_lr_mult(step / args.num_iterations)
    for opt in [adamw, muon]:
        for g in opt.param_groups: g['lr'] = g['initial_lr'] * lr_mult if 'initial_lr' in g else g['lr']
    muon_momentum = get_muon_momentum(step)
    for g in muon.param_groups: g['momentum'] = muon_momentum

    for micro_step in range(grad_accum_steps):
        x, y = next(train_loader)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()

    adamw.step(); muon.step(); torch.cuda.synchronize()
    adamw.zero_grad(); muon.zero_grad()
    nan_params = [(n, p.shape) for n, p in model.named_parameters() if torch.isnan(p).any()]
    if nan_params: print(f"  NAN in params: {nan_params}")

    dt = time.time() - t0
    if step > 10: total_time += dt
    train_loss = loss.item() * grad_accum_steps
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * step / args.num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    mfu = 100 * num_flops_per_token * args.total_batch_size / dt / 989e12
    print(f"step {step:05d} ({pct:5.2f}%) | loss {debiased_loss:.6f} | lrm {lr_mult:.2f} | dt {dt*1000:.2f}ms | tok/s {tok_per_sec:,} | mfu {mfu:.2f}% | time {total_time/60:.2f}m")
    wandb_run.log(dict(step=step, loss=train_loss, lr_mult=lr_mult))
    t0 = time.time()

    if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, val_bpb)
        print(f"step {step:05d} | val_bpb {val_bpb:.4f} | min_val_bpb {min_val_bpb:.4f}")
        wandb_run.log(dict(step=step, val_bpb=val_bpb, min_val_bpb=min_val_bpb))
        model.train()

    if args.save_every > 0 and (step + 1) % args.save_every == 0:
        save_checkpoint(mid_checkpoint_dir, step, model.state_dict(), dict(adamw=adamw.state_dict(), muon=muon.state_dict()), dict(step=step, val_bpb=val_bpb, min_val_bpb=min_val_bpb, config=vars(args), model_config=model_config.__dict__))

final_loss = loss.item() * grad_accum_steps
print(f"Training complete. Final loss: {final_loss:.4f}, min_val_bpb: {min_val_bpb:.4f}")

from nanochat_vl.report import get_report
get_report().log(section="Midtraining", data=[vars(args), dict(num_iterations=args.num_iterations, min_val_bpb=min_val_bpb)])
