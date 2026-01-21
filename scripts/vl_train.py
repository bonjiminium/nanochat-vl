"Minimal single-GPU VLM training script."

import os, time, argparse, logging
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"] = "-dynamo"
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*IndexPutBackward0.*")
import torch
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.vlm import VLM
from nanochat_vl.muon import Muon
from nanochat_vl.common import get_base_dir, DummyWandb
from nanochat_vl.checkpoint_manager import load_model, save_checkpoint
from pathlib import Path
from nanochat_vl.vl_dataloader import vl_data_generator
from tasks.aokvqa import AOKVQA

p = argparse.ArgumentParser()
p.add_argument("--wandb", type=int, default=0)
p.add_argument("--num_steps", type=int, default=1000)
p.add_argument("--device_batch_size", type=int, default=4)
p.add_argument("--target_examples_per_step", type=int, default=32)
p.add_argument("--lr_vision", type=float, default=3e-4)
p.add_argument("--lr_projector", type=float, default=3e-2)
p.add_argument("--lr_lm", type=float, default=3e-4)
p.add_argument("--use_muon", type=int, default=0)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--patch_size", type=int, default=16)
p.add_argument("--vision_dim", type=int, default=256)
p.add_argument("--max_seq_len", type=int, default=512)
p.add_argument("--save_every", type=int, default=500)
p.add_argument("--print_every", type=int, default=1)
p.add_argument("--use_images", type=int, default=1)
p.add_argument("--init_lr_frac", type=float, default=1.0)
args = p.parse_args()

base_dir = Path(get_base_dir())
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

gpt, tokenizer, _ = load_model("sft", device, phase="train")
vlm = VLM(gpt, args.img_size, args.patch_size, args.vision_dim).to(device)
vlm = torch.compile(vlm)

dmodel_lr_scale = (gpt.config.n_embd / 768) ** -0.5
adamw_params = [{'params': list(vlm.vit.parameters()), 'lr': args.lr_vision}, {'params': list(vlm.proj.parameters()), 'lr': args.lr_projector}, {'params': [vlm.gpt.transformer.wte.weight], 'lr': args.lr_lm * 0.1 * dmodel_lr_scale}, {'params': [vlm.gpt.lm_head.weight], 'lr': args.lr_lm * 0.1 * dmodel_lr_scale}]
adamw = torch.optim.AdamW(adamw_params, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

matrix_params = [p for n, p in vlm.gpt.named_parameters() if p.ndim == 2 and 'wte' not in n and 'lm_head' not in n]
if args.use_muon: muon = Muon(matrix_params, lr=args.lr_lm, momentum=0.95)
else: muon = torch.optim.AdamW(matrix_params, lr=args.lr_lm * dmodel_lr_scale, betas=(0.9, 0.95), weight_decay=0.0)

train_ds = AOKVQA(split="train", use_images=bool(args.use_images))
img_token_id = tokenizer.encode_special("<image>")
num_patches = (args.img_size // args.patch_size) ** 2
train_gen = vl_data_generator(train_ds, tokenizer, args.device_batch_size, args.img_size, args.max_seq_len, num_patches, device)

assert args.target_examples_per_step % args.device_batch_size == 0, "target_examples_per_step must be divisible by device_batch_size"
grad_accum_steps = args.target_examples_per_step // args.device_batch_size

def get_lr_mult(step): return args.init_lr_frac * (1.0 - step / args.num_steps)

for opt in [adamw, muon]:
    for group in opt.param_groups: group["initial_lr"] = group["lr"]

wandb_run = DummyWandb() if not args.wandb else __import__('wandb').init(project="nanochat-vl", config=vars(args))

num_flops_per_token = gpt.estimate_flops()
smooth_loss, total_time, ema_beta = 0.0, 0.0, 0.9
tokens_per_step = args.device_batch_size * args.max_seq_len * grad_accum_steps
t0 = time.time()

for step in range(args.num_steps):
    lr_mult = get_lr_mult(step)
    for opt in [adamw, muon]:
        for group in opt.param_groups: group["lr"] = group["initial_lr"] * lr_mult
    adamw.zero_grad()
    muon.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        imgs, x, y = next(train_gen)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = vlm(imgs, x, y, img_token_id=img_token_id)
        loss_accum += loss.item()
        loss.backward()
    loss_accum /= grad_accum_steps
    adamw.step()
    muon.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    total_time += dt
    t0 = time.time()
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_accum
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * step / args.num_steps
    tok_per_sec = int(tokens_per_step / dt)
    mfu = 100 * num_flops_per_token * tokens_per_step / dt / 989e12
    
    if step % args.print_every == 0: print(f"step {step:05d} ({pct:5.2f}%) | loss {debiased_loss:.6f} | lrm {lr_mult:.2f} | vnorm {vlm.vision_norm:.2f} | tnorm {vlm.text_norm:.2f} | dt {dt*1000:.2f}ms | tok/s {tok_per_sec:,} | mfu {mfu:.2f}% | time {total_time/60:.2f}m")
    wandb_run.log(dict(step=step, loss=loss_accum, dt=dt))
    
    if step > 0 and step % args.save_every == 0:
        checkpoint_dir = base_dir / "vl_checkpoints" / f"d{gpt.config.n_layer}"
        model_data = {k.removeprefix("_orig_mod."): v for k, v in vlm.state_dict().items()}
        meta = dict(step=step, loss=loss_accum, vlm_config=dict(img_size=args.img_size, patch_size=args.patch_size, vision_dim=args.vision_dim), model_config=vars(gpt.config))
        save_checkpoint(checkpoint_dir, step, model_data, None, meta)

checkpoint_dir = base_dir / "vl_checkpoints" / f"d{gpt.config.n_layer}"
model_data = {k.removeprefix("_orig_mod."): v for k, v in vlm.state_dict().items()}
meta = dict(step=step, loss=loss_accum, vlm_config=dict(img_size=args.img_size, patch_size=args.patch_size, vision_dim=args.vision_dim), model_config=vars(gpt.config))
save_checkpoint(checkpoint_dir, args.num_steps, model_data, None, meta)
print("VLM training complete!")

from nanochat_vl.report import get_report
get_report().log(section="VL Training", data=[
    {"num_steps": args.num_steps},
    {"device_batch_size": args.device_batch_size},
    {"target_examples_per_step": args.target_examples_per_step},
    {"lr_vision": args.lr_vision},
    {"lr_projector": args.lr_projector},
    {"lr_lm": args.lr_lm},
    {"final_loss": loss_accum},
])
