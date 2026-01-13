import modal, subprocess

app = modal.App("nanochat-vl-speedrun")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2")
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

@app.function(image=image, timeout=60, gpu="L4")
def test_muon():
    import torch
    from nanochat_vl.gpt import GPT, GPTConfig
    from nanochat_vl.muon import Muon
    cfg = GPTConfig(seq_len=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=256)
    model = GPT(cfg).cuda().bfloat16()
    model.init_weights()
    opt = Muon(model.parameters(), lr=0.01)
    losses = []
    for i in range(10):
        x = torch.randint(0, cfg.vocab_size, (4, 32)).cuda()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): loss = model(x, x)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        print(f"step {i}: {loss.item():.4f}")
    print(f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

@app.function(image=image, timeout=180, gpu="L4")
def test_train():
    import subprocess
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--embedding_lr=0.003", "--unembedding_lr=0.0001", "--matrix_lr=0.0003", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=10", "--core_max_per_task=3"], check=True)

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

@app.function(image=image, timeout=180, gpu="L4")
def test_bpb():
    import subprocess, torch
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    from nanochat_vl.gpt import GPT, GPTConfig
    from nanochat_vl.dataloader import data_loader
    from nanochat_vl.tokenizer import get_token_bytes
    from nanochat_vl.loss_eval import evaluate_bpb
    cfg = GPTConfig(seq_len=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=128, vocab_size=4096)
    model = GPT(cfg).cuda().bfloat16()
    model.init_weights()
    token_bytes = get_token_bytes(device="cuda")
    loader = data_loader(B=4, T=64, split="train", device="cuda")
    bpb = evaluate_bpb(model, loader, steps=10, token_bytes=token_bytes)
    print(f"BPB (untrained): {bpb:.4f}")

@app.function(image=image, timeout=600, gpu="L4")
def test_core():
    import subprocess, torch
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    from nanochat_vl.gpt import GPT, GPTConfig
    from nanochat_vl.tokenizer import get_tokenizer
    from nanochat_vl.base_eval import evaluate_model
    cfg = GPTConfig(seq_len=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=128, vocab_size=4096)
    model = GPT(cfg).cuda().bfloat16()
    model.init_weights()
    model.eval()
    tokenizer = get_tokenizer()
    results = evaluate_model(model, tokenizer, 'cuda', max_per_task=5)
    print(f"CORE metric (untrained, 5/task): {results['core_metric']:.4f}")

@app.local_entrypoint()
def main(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536, test: str = ""):
    if test == "gpt": return test_gpt.remote()
    if test == "muon": return test_muon.remote()
    if test == "train": return test_train.remote()
    if test == "dataloader": return test_dataloader.remote()
    if test == "bpb": return test_bpb.remote()
    if test == "core": return test_core.remote()
    from nanochat_vl.report import get_git_info, get_bloat_info
    git_info, bloat_info = get_git_info(), get_bloat_info()
    speedrun.remote(n_shards, max_chars, vocab_size, git_info, bloat_info)
