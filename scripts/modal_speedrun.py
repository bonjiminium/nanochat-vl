import modal, subprocess, os, glob

app = modal.App("nanochat-vl-speedrun")
volume = modal.Volume.from_name("nanochat-vl-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2", "wandb", "datasets", "pillow")
    .env({"PYTHONUNBUFFERED": "1"}) # so training scripts log immediately
    .add_local_dir('.', '/root')
)

CHECKPOINT_DIRS = dict(base="base_checkpoints", mid="mid_checkpoints", sft="chatsft_checkpoints", vl="vl_checkpoints")

def setup_env():
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

def has_checkpoint(stage):
    from nanochat_vl.common import get_base_dir
    d = os.path.join(get_base_dir(), CHECKPOINT_DIRS[stage])
    return bool(glob.glob(os.path.join(d, "d*", "model_*.pt")))

def ensure_tokenizer():
    from nanochat_vl.common import get_base_dir
    from pathlib import Path
    if (Path(get_base_dir()) / "tokenizer").exists(): return
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", "2"], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)

def ensure_base():
    ensure_tokenizer()
    if has_checkpoint("base"): return
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--aspect_ratio=64", "--head_dim=64", "--max_seq_len=1024", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=4096", "--num_iterations=20", "--warmup_ratio=0.1", "--warmdown_ratio=0.2", "--eval_every=-1", "--core_metric_every=-1", "--save_every=20"], check=True)

def ensure_mid():
    ensure_base()
    if has_checkpoint("mid"): return
    subprocess.run(["python", "-m", "scripts.mid_train", "--num_iterations=10", "--device_batch_size=4", "--total_batch_size=4096", "--max_seq_len=1024", "--eval_every=-1", "--save_every=10"], check=True)

def ensure_sft():
    ensure_mid()
    if has_checkpoint("sft"): return
    subprocess.run(["python", "-m", "scripts.chat_sft", "--num_iterations=10", "--device_batch_size=4", "--target_examples_per_step=32", "--max_seq_len=1024", "--eval_every=-1", "--eval_steps=4", "--save_every=10"], check=True)

def ensure_vl():
    ensure_sft()
    if has_checkpoint("vl"): return
    subprocess.run(["python", "-m", "scripts.vl_train", "--num_steps=20", "--batch_size=2", "--grad_accum=1", "--use_images=1", "--print_every=10"], check=True)

@app.function(image=image, timeout=7200, gpu="H100", secrets=[modal.Secret.from_name("wandb-secret")])
def speedrun(run: str = "dummy", git_info: dict = None, bloat_info: dict = None):
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

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume})
def test_base():
    setup_env()
    ensure_tokenizer()
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=2", "--aspect_ratio=64", "--head_dim=64", "--max_seq_len=1024", "--vocab_size=4096", "--device_batch_size=4", "--total_batch_size=4096", "--num_iterations=20", "--warmup_ratio=0.1", "--warmdown_ratio=0.2", "--eval_every=5", "--eval_tokens=4096", "--core_metric_every=-1", "--save_every=20", "--resume_from=-2"], check=True)
    volume.commit()

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume})
def test_mid():
    setup_env()
    ensure_base()
    subprocess.run(["python", "-m", "scripts.mid_train", "--num_iterations=10", "--device_batch_size=4", "--total_batch_size=4096", "--max_seq_len=1024", "--eval_every=5", "--eval_tokens=4096", "--save_every=10"], check=True)
    volume.commit()

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_sft():
    setup_env()
    ensure_mid()
    subprocess.run(["python", "-m", "scripts.chat_sft", "--num_iterations=10", "--device_batch_size=4", "--target_examples_per_step=32", "--max_seq_len=1024", "--eval_every=5", "--eval_steps=4", "--save_every=10"], check=True)
    volume.commit()

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_vl():
    setup_env()
    ensure_sft()
    subprocess.run(["python", "-m", "scripts.vl_train", "--num_steps=20", "--batch_size=2", "--grad_accum=2", "--max_seq_len=1024", "--use_muon=0", "--print_every=5"], check=True)
    volume.commit()

@app.function(image=image, timeout=300, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_temp():
    setup_env()
    ensure_sft()
    import subprocess
    print("Training VL with 224x224 images...")
    subprocess.run(["python", "-m", "scripts.vl_train", "--num_steps=20", "--device_batch_size=2", "--target_examples_per_step=8", "--use_images=1", "--print_every=5"], check=True)
    print("Testing vl_eval with verbose output (5 examples)...")
    result = subprocess.run(["python", "-m", "scripts.vl_eval", "--task-name=AOKVQA", "--max-problems=5", "--use-images=1"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("vl_eval failed")
    print("SUCCESS: vl_eval works")



@app.local_entrypoint()
def main(test: str = "", run: str = "dummy"):
    if test == "base": return test_base.remote()
    if test == "mid": return test_mid.remote()
    if test == "sft": return test_sft.remote()
    if test == "vl": return test_vl.remote()
    if test == "temp": return test_temp.remote()
    from nanochat_vl.report import get_git_info, get_bloat_info
    speedrun.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info())
