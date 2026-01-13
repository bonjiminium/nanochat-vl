import modal, subprocess

app = modal.App("nanochat-vl-speedrun")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil")
    .add_local_dir('.', '/root')
)

@app.function(image=image, timeout=3600)
def speedrun(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536, git_info: dict = None, bloat_info: dict = None):
    import os
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    gpu_info = get_gpu_info()
    report.reset(git_info, bloat_info, gpu_info, get_system_info(), estimate_cost(gpu_info), get_dep_count())
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", str(n_shards)], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars", str(max_chars), "--vocab_size", str(vocab_size)], check=True)
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=120, gpu="L4")
def test_dataloader():
    import subprocess
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    from nanochat_vl.dataloader import data_loader
    loader = data_loader(B=2, T=64, split="train", device="cuda")
    x, y = next(loader)
    print(f"x: {x.shape}, y: {y.shape}, x[0,:10]: {x[0,:10].tolist()}")

@app.function(image=image, timeout=60, gpu="L4")
def test_gpt():
    import torch
    from nanochat_vl.gpt import GPT, GPTConfig
    cfg = GPTConfig(seq_len=512, n_layer=4, n_head=4, n_kv_head=4, n_embd=256)
    model = GPT(cfg).cuda().bfloat16()
    model.init_weights()
    x = torch.randint(0, cfg.vocab_size, (2, 64)).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, x)
    print(f"Params: {model.num_params():,}, Loss: {loss.item():.4f}")

@app.local_entrypoint()
def main(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536, test: str = ""):
    if test == "gpt": return test_gpt.remote()
    if test == "dataloader": return test_dataloader.remote()
    from nanochat_vl.report import get_git_info, get_bloat_info
    git_info, bloat_info = get_git_info(), get_bloat_info()
    speedrun.remote(n_shards, max_chars, vocab_size, git_info, bloat_info)
