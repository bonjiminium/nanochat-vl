"VL Ablation: compare training with vs without images on A-OKVQA."

import modal, subprocess, os, glob, time, csv, sys

app = modal.App("vl-ablation")
volume = modal.Volume.from_name("vl-ablation-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2", "wandb", "datasets", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir('.', '/root')
)

CHECKPOINT_DIRS = dict(base="base_checkpoints", mid="mid_checkpoints", sft="chatsft_checkpoints", vl="vl_checkpoints")
TOK = dict(max_chars=2_000_000_000, vocab_size=65536)
BASE = dict(depth=16, vocab_size=65536, device_batch_size=32)
MID = dict(num_iterations=2187, device_batch_size=32, total_batch_size=524288, eval_every=150)
SFT = dict(num_iterations=417, device_batch_size=32, target_examples_per_step=32, max_seq_len=1024)
VL = dict(num_steps=2000, device_batch_size=32, target_examples_per_step=32, max_seq_len=1024, print_every=50)

def setup_env():
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

def run(cmd): subprocess.run(cmd, check=True)
def run_script(script, cfg): run(["python", "-u", "-m", script] + [f"--{k}={v}" for k,v in cfg.items()])

def has_checkpoint(phase): return bool(glob.glob(f"/data/{CHECKPOINT_DIRS[phase]}/d*/model_*.pt"))

def ensure_tokenizer():
    from pathlib import Path
    tok_path = Path("/data/tokenizer/tokenizer.pkl")
    if tok_path.exists():
        import pickle
        with open(tok_path, "rb") as f: enc = pickle.load(f)
        if enc.n_vocab == TOK["vocab_size"]: return
    run(["python", "-m", "nanochat_vl.dataset", "-n", "240"])
    run_script("scripts.tok_train", TOK)

def ensure_base():
    ensure_tokenizer()
    if has_checkpoint("base"): return
    run_script("scripts.base_train", BASE)

def ensure_mid():
    ensure_base()
    if has_checkpoint("mid"): return
    run_script("scripts.mid_train", MID)

def ensure_sft():
    ensure_mid()
    if has_checkpoint("sft"): return
    run_script("scripts.chat_sft", SFT)

def train_vl(use_images):
    for p in glob.glob("/data/vl_checkpoints/*"): subprocess.run(["rm", "-rf", p])
    start = time.time()
    run_script("scripts.vl_train", {**VL, "use_images": use_images})
    return time.time() - start

def eval_vl(use_images, max_problems):
    args = ["python", "-m", "scripts.vl_eval", f"--task_name=AOKVQA", f"--use_images={use_images}", f"--max_problems={max_problems}"]
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.split("\n"):
        if "accuracy:" in line.lower(): return float(line.split()[-1].replace("%", ""))
    return 0.0

@app.function(image=image, timeout=28800, gpu="H100", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")])
def run_ablation(series_name: str = "ablation", eval_max: int = 500):
    import wandb, shutil
    setup_env()
    for f in os.listdir("/data"):
        p = f"/data/{f}"
        if os.path.isdir(p): shutil.rmtree(p)
        else: os.remove(p)
    print("Wiped volume, starting fresh...")
    ensure_sft()
    volume.commit()
    
    print(f"\n{'='*60}\nEvaluating SFT model\n{'='*60}")
    run(["python", "-m", "scripts.chat_eval", "-i", "sft", f"--max_problems={eval_max}"])
    
    wandb.init(project="nanochat-vl", name=f"{series_name}", config=dict(series=series_name, eval_max=eval_max, **VL))
    
    results = []
    for use_images in [1, 0]:
        config = "images" if use_images else "no_images"
        print(f"\n{'='*60}\nTraining: {config}\n{'='*60}")
        train_time = train_vl(use_images)
        print(f"\n{'='*60}\nEvaluating: {config}\n{'='*60}")
        val_acc = eval_vl(use_images, eval_max)
        results.append(dict(config=config, val_acc=val_acc, train_time_min=train_time/60))
        print(f"{config}: val_acc={val_acc:.2f}%, train_time={train_time/60:.1f}min")
        volume.commit()
    
    print(f"\n{'='*60}\nRESULTS: {series_name}\n{'='*60}")
    print(f"{'config':<12} {'val_acc':>10} {'train_min':>10}")
    for r in results: print(f"{r['config']:<12} {r['val_acc']:>9.2f}% {r['train_time_min']:>10.1f}")
    
    diff = results[0]['val_acc'] - results[1]['val_acc']
    print(f"\nDelta (images - no_images): {diff:+.2f}%")
    verdict = "PASS: images help" if diff >= 3 else "FAIL: images don't help enough" if diff > 0 else "FAIL: images hurt"
    print(verdict)
    
    wandb.log(dict(images_acc=results[0]['val_acc'], no_images_acc=results[1]['val_acc'], delta=diff))
    wandb.finish()
    volume.commit()

@app.local_entrypoint()
def main(series: str = "ablation", eval_max: int = 500):
    run_ablation.remote(series, eval_max)
