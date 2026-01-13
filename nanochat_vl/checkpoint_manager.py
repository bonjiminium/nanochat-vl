"Utilities for saving and loading model/optim/state checkpoints."
import os, glob, json, torch
from nanochat_vl.common import get_base_dir
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.tokenizer import get_tokenizer

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model_data, os.path.join(checkpoint_dir, f"model_{step:06d}.pt"))
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"), "w") as f: json.dump(meta_data, f, indent=2)
    if optimizer_data is not None: torch.save(optimizer_data, os.path.join(checkpoint_dir, f"optim_{step:06d}.pt"))
    print(f"Saved checkpoint at step {step} to {checkpoint_dir}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    model_data = torch.load(os.path.join(checkpoint_dir, f"model_{step:06d}.pt"), map_location=device)
    with open(os.path.join(checkpoint_dir, f"meta_{step:06d}.json")) as f: meta_data = json.load(f)
    optimizer_data = torch.load(os.path.join(checkpoint_dir, f"optim_{step:06d}.pt"), map_location=device) if load_optimizer else None
    return model_data, optimizer_data, meta_data

def find_last_checkpoint_dir(base_checkpoint_dir):
    subdirs = glob.glob(os.path.join(base_checkpoint_dir, "d*"))
    for d in sorted(subdirs, reverse=True):
        if glob.glob(os.path.join(d, "model_*.pt")): return d
    return None

def find_last_step(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not files: return None
    return max(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in files)

def build_model(checkpoint_dir, step, device, phase="eval"):
    if step is None: step = find_last_step(checkpoint_dir)
    if step is None: raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg = GPTConfig(**meta_data["model_config"])
    with torch.device("meta"): model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    if phase == "eval": model.eval()
    else: model.train()
    return model, get_tokenizer(), meta_data

def load_model(source, device, phase="eval", step=None):
    base_dir = get_base_dir()
    base_checkpoint_dir = os.path.join(base_dir, {"base": "base_checkpoints", "mid": "mid_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[source])
    checkpoint_dir = find_last_checkpoint_dir(base_checkpoint_dir)
    if checkpoint_dir is None: raise FileNotFoundError(f"No checkpoints in {base_checkpoint_dir}")
    return build_model(checkpoint_dir, step, device, phase)
