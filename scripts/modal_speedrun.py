import modal, subprocess

app = modal.App("nanochat-vl-speedrun")
volume = modal.Volume.from_name("nanochat-vl-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2", "wandb", "datasets", "pillow")
    .add_local_dir('.', '/root')
)

@app.function(image=image, timeout=7200, gpu="H100", secrets=[modal.Secret.from_name("wandb-secret")])
def speedrun(run: str = "dummy", git_info: dict = None, bloat_info: dict = None):
    import os
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "240"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=2000000000", "--vocab_size=32768"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_eval"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=10", "--vocab_size=32768", "--device_batch_size=32", "--eval_every=100", "--eval_tokens=10485760", "--core_metric_every=100", f"--run={run}"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_loss", "--eval_tokens=10485760", "--device_batch_size=32"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_eval", "--max_problems=500", "--batch_size=32"], check=True)
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

@app.function(image=image, timeout=300, gpu="L4", secrets=[modal.Secret.from_name("wandb-secret")])
def test_train(run: str = "dummy", git_info: dict = None, bloat_info: dict = None):
    import os, subprocess
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--embedding_lr=0.003", "--unembedding_lr=0.0001", "--matrix_lr=0.0003", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=10", "--core_max_per_task=3", f"--run={run}"], check=True)
    subprocess.run(["python", "-m", "scripts.base_loss", "--eval_tokens=1024", "--device_batch_size=4"], check=True)
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=600, gpu="L4", secrets=[modal.Secret.from_name("wandb-secret")])
def test_train_med(run: str = "dummy", git_info: dict = None, bloat_info: dict = None):
    import os, subprocess
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=12", "--max_seq_len=256", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--embedding_lr=0.3", "--unembedding_lr=0.004", "--matrix_lr=0.02", f"--run={run}"], check=True)
    subprocess.run(["python", "-m", "scripts.base_loss", "--eval_tokens=1024", "--device_batch_size=4"], check=True)
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=7200, gpu="H100", secrets=[modal.Secret.from_name("wandb-secret")])
def test_speedrun(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, num_iterations: int = 1400):
    import os, subprocess
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "240"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=2000000000", "--vocab_size=32768"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_eval"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=10", "--vocab_size=32768", "--device_batch_size=32", "--eval_every=100", "--eval_tokens=10485760", "--core_metric_every=-1", f"--run={run}"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_loss", "--eval_tokens=10485760", "--device_batch_size=32"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_eval", "--max_problems=500", "--batch_size=32"], check=True)
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
    import subprocess
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--save_every=20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_loss", "--eval_tokens=1024", "--device_batch_size=4"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_eval", "--max_problems=20", "--batch_size=8"], check=True)

@app.function(image=image, timeout=180, gpu="L4")
def test_checkpoint():
    import subprocess, os
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    print("=== Training 5 steps, saving every 2 ===")
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=5", "--warmup_iters=1", "--cooldown_iters=1", "--embedding_lr=0.003", "--unembedding_lr=0.0001", "--matrix_lr=0.0003", "--eval_every=100", "--core_metric_every=-1", "--save_every=2"], check=True)
    print("=== Resuming from step 4, training to 8 ===")
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=8", "--warmup_iters=1", "--cooldown_iters=1", "--embedding_lr=0.003", "--unembedding_lr=0.0001", "--matrix_lr=0.0003", "--eval_every=100", "--core_metric_every=-1", "--save_every=2", "--resume_from=-1"], check=True)
    print("=== Checkpoint test passed ===")

@app.function(image=image, timeout=180)
def test_report(git_info: dict = None, bloat_info: dict = None):
    import time, os
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count, extract_timestamp
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    header_path = os.path.join(get_base_dir(), "report", "header.md")
    print(f"Header content:\n{open(header_path).read()}")
    print("Sleeping 90 seconds...")
    time.sleep(90)
    report.log(section="Base model training", data=[{"test": "value"}])
    section_path = os.path.join(get_base_dir(), "report", "base-model-training.md")
    print(f"Section content:\n{open(section_path).read()}")
    with open(header_path) as f: start_time = extract_timestamp(f.read(), "Run started: ")
    with open(section_path) as f: end_time = extract_timestamp(f.read(), "timestamp: ")
    print(f"Extracted start_time: {start_time}")
    print(f"Extracted end_time: {end_time}")
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=600, gpu="L4", secrets=[modal.Secret.from_name("huggingface-secret")])
def test_pipeline(git_info: dict = None, bloat_info: dict = None):
    import os, subprocess
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_eval"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--save_every=20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_loss", "--eval_tokens=1024", "--device_batch_size=4"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_eval", "--max_problems=20", "--batch_size=8"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_iterations=10", "--device_batch_size=4", "--max_seq_len=64", "--eval_every=5", "--save_every=10"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.chat_eval", "-i", "mid", "-x", "20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.chat_sft", "--num_iterations=10", "--device_batch_size=4", "--max_seq_len=64", "--eval_every=5", "--save_every=10"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.chat_eval", "-i", "sft", "-x", "20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.vl_train", "--num_steps=50", "--batch_size=2", "--grad_accum=2", "--max_seq_len=64", "--use_muon=0", "--print_every=10"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.vl_eval", "--max-problems=50", "--batch-size=4"], check=True)
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=600, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_tokenizer(git_info: dict = None, bloat_info: dict = None):
    import os, subprocess
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "16"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=4000000000", "--vocab_size=65536"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_eval"], check=True)
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())

@app.function(image=image, timeout=60, gpu="L4")
def test_mid_dataloader():
    import subprocess
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    from tasks.smoltalk import SmolTalk
    from nanochat_vl.tokenizer import get_tokenizer
    from nanochat_vl.mid_dataloader import mid_data_generator
    ds = SmolTalk(split="train", stop=100)
    tokenizer = get_tokenizer()
    gen = mid_data_generator(ds, tokenizer, batch_size=4, seq_len=64, device="cuda")
    x, y = next(gen)
    print(f"x: {x.shape}, y: {y.shape}, x.dtype: {x.dtype}, y.dtype: {y.dtype}")
    print(f"x[0,:10]: {x[0,:10].tolist()}")
    print(f"y[0,:10]: {y[0,:10].tolist()}")
    x2, y2 = next(gen)
    print(f"Second batch x[0,:10]: {x2[0,:10].tolist()}")

@app.function(image=image, timeout=60, gpu="L4")
def test_mid(git_info: dict = None, bloat_info: dict = None):
    import subprocess
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--aspect_ratio=64", "--head_dim=64", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=256", "--num_iterations=20", "--warmup_ratio=0.1", "--warmdown_ratio=0.2", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--save_every=20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_iterations=10", "--device_batch_size=4", "--max_seq_len=64", "--eval_every=5", "--save_every=10"], check=True)
    report.generate()

@app.function(image=image, timeout=600, gpu="L4", secrets=[modal.Secret.from_name("huggingface-secret")])
def test_chat_eval():
    import subprocess
    subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--embedding_lr=0.003", "--unembedding_lr=0.0001", "--matrix_lr=0.0003", "--eval_every=-1", "--core_metric_every=-1", "--save_every=20"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.chat_eval", "-i", "base", "-x", "20"], check=True)

@app.function(image=image, timeout=60, gpu="L4", secrets=[modal.Secret.from_name("huggingface-secret")])
def test_mmlu():
    from tasks.mmlu import MMLU
    ds = MMLU(subset="all", split="test", stop=5)
    print(f"len(ds) = {len(ds)}")
    ex = ds[0]
    print(f"Example: {ex}")

@app.function(image=image, timeout=60, gpu="L4")
def test_smoltalk():
    import subprocess
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)
    from tasks.smoltalk import SmolTalk
    from nanochat_vl.tokenizer import get_tokenizer
    ds = SmolTalk(split="train", stop=5)
    print(f"len(ds) = {len(ds)}")
    ex = ds[0]
    print(f"Example keys: {ex.keys()}")
    print(f"Num messages: {len(ex['messages'])}")
    print(f"First message role: {ex['messages'][0]['role']}")
    # Test render_conversation
    tokenizer = get_tokenizer()
    ids, mask = tokenizer.render_conversation(ex)
    print(f"ids length: {len(ids)}, mask length: {len(mask)}")
    print(f"mask sum (supervised tokens): {sum(mask)}")
    print(f"First 20 ids: {ids[:20]}")
    print(f"First 20 mask: {mask[:20]}")

@app.function(image=image, timeout=600, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_vlm():
    import os, subprocess
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    from pathlib import Path
    from nanochat_vl.common import get_base_dir
    base = Path(get_base_dir())
    if not (base / "tokenizer").exists():
        print("Training tokenizer...")
        subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--vocab_size", "4096", "--max_chars", "10000000"], check=True)
    if not (base / "base_data").exists():
        subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    import shutil
# RESTORE: uncomment these to skip retraining base/mid
    if not (base / "base_checkpoints").exists():
    # if (base / "base_checkpoints").exists(): shutil.rmtree(base / "base_checkpoints")
        print("Training base model...")
        subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--embedding_lr=0.0003", "--unembedding_lr=0.00001", "--matrix_lr=0.00003", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--save_every=20"], check=True)
    # RESTORE: uncomment to skip retraining mid
    if not (base / "mid_checkpoints").exists():
    # if (base / "mid_checkpoints").exists(): shutil.rmtree(base / "mid_checkpoints")
        print("Training mid model...")
        subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_iterations=12", "--device_batch_size=4", "--max_seq_len=64", "--eval_every=5", "--save_every=12", "--matrix_lr=0.0002", "--use_muon=0"], check=True)
    # # RESTORE: uncomment to skip retraining sft
    # if not (base / "chatsft_checkpoints").exists():
    # RESTORE: uncomment to skip retraining SFT
    if not (base / "chatsft_checkpoints").exists():
    # if (base / "chatsft_checkpoints").exists(): shutil.rmtree(base / "chatsft_checkpoints")
        print("Training SFT model...")
        subprocess.run(["python", "-u", "-m", "scripts.chat_sft", "--num_iterations=10", "--device_batch_size=4", "--max_seq_len=256", "--eval_every=5", "--save_every=10", "--use_muon=0"], check=True)
    
    import torch
    from nanochat_vl.checkpoint_manager import load_model
    from nanochat_vl.vlm import VLM
    from nanochat_vl.vl_dataloader import vl_data_generator
    model, tokenizer, meta = load_model("sft", "cuda")
    vlm = VLM(model, img_size=64, patch_size=8, vision_dim=256).cuda()
    test_data = [("a red apple", None), ("a yellow banana", None), ("a dog", None)]
    gen = vl_data_generator(test_data, tokenizer, batch_size=2, img_size=64, max_seq_len=32, device="cuda")
    imgs, inputs, targets = next(gen)
    loss = vlm(imgs, inputs, targets)
    print(f"VLM forward pass successful! Loss: {loss.item():.3f}")

@app.function(image=image, timeout=900, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_vl_train():
    import os, subprocess
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    from pathlib import Path
    from nanochat_vl.common import get_base_dir
    base = Path(get_base_dir())
    if not (base / "tokenizer").exists(): subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--vocab_size", "4096", "--max_chars", "10000000"], check=True)
    if not (base / "base_data").exists(): subprocess.run(["python", "-u", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    if not (base / "base_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.base_train", "--depth=2", "--n_embd=128", "--n_head=2", "--max_seq_len=64", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=16", "--num_iterations=20", "--warmup_iters=2", "--cooldown_iters=2", "--embedding_lr=0.0003", "--unembedding_lr=0.00001", "--matrix_lr=0.00003", "--eval_every=5", "--eval_tokens=1024", "--core_metric_every=-1", "--save_every=20"], check=True)
    if not (base / "mid_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_iterations=12", "--device_batch_size=4", "--max_seq_len=64", "--eval_every=5", "--save_every=12", "--matrix_lr=0.0002", "--use_muon=0"], check=True)
    if not (base / "chatsft_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.chat_sft", "--num_iterations=10", "--device_batch_size=4", "--max_seq_len=256", "--eval_every=5", "--save_every=10", "--use_muon=0"], check=True)
    print("=== Running VL train for 20 steps ===")
    subprocess.run(["python", "-u", "-m", "scripts.vl_train", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--max_seq_len=64", "--use_muon=0", "--print_every=1"], check=True)
    volume.commit()

@app.function(image=image, timeout=900, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_vl_eval():
    import os, subprocess
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    from pathlib import Path
    from nanochat_vl.common import get_base_dir
    base = Path(get_base_dir())
    if not (base / "tokenizer").exists(): subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--vocab_size", "4096", "--max_chars", "10000000"], check=True)
    if not (base / "base_data").exists(): subprocess.run(["python", "-u", "-m", "nanochat_vl.data", "--n_shards=1", "--max_chars=10000000"], check=True)
    if not (base / "base_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.base_train", "--num_steps=20", "--warmdown_steps=5", "--batch_size=2", "--grad_accum=2"], check=True)
    if not (base / "mid_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--use_muon=0"], check=True)
    if not (base / "chatsft_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.chat_sft", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--use_muon=0"], check=True)
    if not (base / "vl_checkpoints").exists():
        print("=== No VLM checkpoint, running VL train first ===")
        subprocess.run(["python", "-u", "-m", "scripts.vl_train", "--num_steps=50", "--batch_size=2", "--grad_accum=2", "--max_seq_len=64", "--use_muon=0", "--print_every=10"], check=True)
    print("=== Running VL eval on ScienceQA (50 problems) ===")
    subprocess.run(["python", "-u", "-m", "scripts.vl_eval", "--max-problems=50", "--batch-size=4"], check=True)

@app.function(image=image, timeout=1800, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_vl_report():
    import os, subprocess
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    from pathlib import Path
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report
    base = Path(get_base_dir())
    if not (base / "tokenizer").exists(): subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--vocab_size", "4096", "--max_chars", "10000000"], check=True)
    if not (base / "base_data").exists(): subprocess.run(["python", "-u", "-m", "nanochat_vl.data", "--n_shards=1", "--max_chars=10000000"], check=True)
    if not (base / "base_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.base_train", "--num_steps=20", "--warmdown_steps=5", "--batch_size=2", "--grad_accum=2"], check=True)
    if not (base / "mid_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.mid_train", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--use_muon=0"], check=True)
    if not (base / "chatsft_checkpoints").exists(): subprocess.run(["python", "-u", "-m", "scripts.chat_sft", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--use_muon=0"], check=True)
    if not (base / "vl_checkpoints").exists():
        subprocess.run(["python", "-u", "-m", "scripts.vl_train", "--num_steps=50", "--batch_size=2", "--grad_accum=2", "--max_seq_len=64", "--use_muon=0", "--print_every=10"], check=True)
    subprocess.run(["python", "-u", "-m", "scripts.vl_eval", "--max-problems=50", "--batch-size=4"], check=True)
    report = get_report()
    report.generate()
    print(open(base / "report" / "report.md").read())

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume})
def test_volume():
    import os
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    from pathlib import Path
    from nanochat_vl.common import get_base_dir
    tok_path = Path(get_base_dir()) / "tokenizer"
    if tok_path.exists():
        print(f"Tokenizer found at {tok_path}, loading...")
        from nanochat_vl.tokenizer import get_tokenizer
        tok = get_tokenizer()
        print(f"Loaded tokenizer with vocab_size={tok.get_vocab_size()}")
    else:
        print(f"No tokenizer at {tok_path}, training...")
        subprocess.run(["python", "-u", "-m", "scripts.tok_train", "--vocab_size", "4096", "--max_chars", "10000000"], check=True)
        print("Saved tokenizer to volume")
    volume.commit()

@app.local_entrypoint()
def main(test: str = "", run: str = "dummy", num_iterations: int = 0):
    if test == "vlm": return test_vlm.remote()
    if test == "vl_train": return test_vl_train.remote()
    if test == "volume": return test_volume.remote()
    if test == "vl_eval": return test_vl_eval.remote()
    if test == "vl_report": return test_vl_report.remote()
    if test == "chat_eval": return test_chat_eval.remote()
    if test == "mmlu": return test_mmlu.remote()
    if test == "smoltalk": return test_smoltalk.remote()
    if test == "mid_dataloader": return test_mid_dataloader.remote()
    if test == "mid":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_mid.remote(git_info=get_git_info(), bloat_info=get_bloat_info())
    if test == "pipeline":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_pipeline.remote(git_info=get_git_info(), bloat_info=get_bloat_info())
    if test == "gpt": return test_gpt.remote()
    if test == "muon": return test_muon.remote()
    if test == "train":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_train.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info())
    if test == "train_med":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_train_med.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info())
    if test == "speedrun":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_speedrun.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info(), num_iterations=num_iterations or 1400)
    if test == "dataloader": return test_dataloader.remote()
    if test == "bpb": return test_bpb.remote()
    if test == "core": return test_core.remote()
    if test == "checkpoint": return test_checkpoint.remote()
    if test == "report":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_report.remote(git_info=get_git_info(), bloat_info=get_bloat_info())
    if test == "tokenizer":
        from nanochat_vl.report import get_git_info, get_bloat_info
        return test_tokenizer.remote(git_info=get_git_info(), bloat_info=get_bloat_info())
    from nanochat_vl.report import get_git_info, get_bloat_info
    speedrun.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info())
