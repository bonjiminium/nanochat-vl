"Minimal single-GPU chat SFT script."

import os, time, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.common import get_base_dir, DummyWandb
from nanochat_vl.tokenizer import get_tokenizer, get_token_bytes
from nanochat_vl.checkpoint_manager import save_checkpoint, load_model
from nanochat_vl.sft_dataloader import sft_data_generator
from tasks.smoltalk import SmolTalk
from tasks.arc import ARC
from tasks.common import TaskMixture

parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--device_batch_size', type=int, default=4)
parser.add_argument('--target_examples_per_step', type=int, default=32)
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--embedding_lr', type=float, default=0.002)
parser.add_argument('--unembedding_lr', type=float, default=0.0004)
parser.add_argument('--matrix_lr', type=float, default=0.002)

parser.add_argument('--cooldown_iters', type=int, default=20)
parser.add_argument('--eval_every', type=int, default=100)
parser.add_argument('--eval_steps', type=int, default=100)
parser.add_argument('--run', type=str, default="dummy")
parser.add_argument('--save_every', type=int, default=-1)
parser.add_argument('--mid_step', type=int, default=-1)
args = parser.parse_args()

device = "cuda"
base_dir = get_base_dir()

model, _, meta = load_model("mid", device, phase="train", step=args.mid_step if args.mid_step >= 0 else None)
model_config = GPTConfig(**meta["model_config"])
sft_checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", f"d{model_config.n_layer}")
model = model.bfloat16()
# model = torch.compile(model) # doesn't work well with variable-length inputs
print(f"Loaded model: {model.num_params():,} parameters")

tokenizer = get_tokenizer()
train_ds = TaskMixture([SmolTalk(split="train", stop=10000), ARC(subset="ARC-Easy", split="train"), ARC(subset="ARC-Challenge", split="train")])
val_ds = SmolTalk(split="test", stop=1000)
train_gen = sft_data_generator(train_ds, tokenizer, args.device_batch_size, args.max_seq_len, device)
val_gen = sft_data_generator(val_ds, tokenizer, args.device_batch_size, args.max_seq_len, device)

adamw, muon = model.setup_optimizers(embedding_lr=args.embedding_lr, unembedding_lr=args.unembedding_lr, matrix_lr=args.matrix_lr)
optimizers = [adamw, muon]

grad_accum_steps = args.target_examples_per_step // args.device_batch_size
wandb_run = DummyWandb() if args.run == "dummy" else __import__("wandb").init(project="nanochat-vl", name=args.run, config=vars(args))

min_val_loss = float('inf')
num_flops_per_token = model.estimate_flops()
smooth_loss, total_time, ema_beta = 0.0, 0.0, 0.9
t0 = time.time()

for step in range(args.num_iterations):
    model.train()
    for opt in optimizers: opt.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        x, y = next(train_gen)
        if (y != -1).sum().item() == 0: continue
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y) / grad_accum_steps
        loss.backward()

    for opt in optimizers: opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    total_time += dt
    t0 = time.time()

    lr_mult = 1.0 if step >= args.cooldown_iters else min(1.0, (args.num_iterations - step) / args.cooldown_iters)
    for opt in optimizers:
        for g in opt.param_groups: g['lr'] = g['initial_lr'] * lr_mult

    train_loss = loss.item() * grad_accum_steps
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * step / args.num_iterations
    tok_per_sec = int(args.target_examples_per_step * args.max_seq_len / dt)
    mfu = 100 * num_flops_per_token * args.target_examples_per_step * args.max_seq_len / dt / 989e12
    print(f"step {step:05d} ({pct:5.2f}%) | loss {debiased_loss:.6f} | lrm {lr_mult:.2f} | dt {dt*1000:.2f}ms | tok/s {tok_per_sec:,} | mfu {mfu:.2f}% | time {total_time/60:.2f}m")
    wandb_run.log(dict(step=step, loss=train_loss))

    if args.eval_every > 0 and step % args.eval_every == 0:
        model.eval()
        eval_steps = args.eval_steps
        with torch.no_grad():
            val_loss_sum, val_count = 0.0, 0
            for i in range(eval_steps):
                x, y = next(val_gen)
                if (y != -1).sum().item() == 0: continue
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    val_loss_sum += model(x, y).item()
                    val_count += 1
            val_loss = val_loss_sum / max(val_count, 1)
        if val_loss < min_val_loss: min_val_loss = val_loss
        print(f"step {step:05d} | val_loss {val_loss:.4f} | min_val_loss {min_val_loss:.4f}")
        wandb_run.log(dict(step=step, val_loss=val_loss, min_val_loss=min_val_loss))
        model.train()

    if args.save_every > 0 and (step + 1) % args.save_every == 0:
        os.makedirs(sft_checkpoint_dir, exist_ok=True)
        save_checkpoint(sft_checkpoint_dir, step, model.state_dict(), dict(adamw=adamw.state_dict(), muon=muon.state_dict()), dict(step=step, min_val_loss=min_val_loss, config=vars(args), model_config=model_config.__dict__))
        print(f"Saved checkpoint at step {step} to {sft_checkpoint_dir}")

final_loss = loss.item() * grad_accum_steps
print(f"Training complete. Final loss: {final_loss:.4f}, min_val_loss: {min_val_loss:.4f}")

from nanochat_vl.report import get_report
get_report().log(section="Chat SFT", data=[vars(args), dict(num_iterations=args.num_iterations, min_val_loss=min_val_loss)])
