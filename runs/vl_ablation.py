"VL Ablation: compare training with vs without images on A-OKVQA."

import modal, subprocess, os, glob, time, csv, sys

app = modal.App("vl-ablation")
volume = modal.Volume.from_name("vl-ablation-data", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2", "wandb", "datasets", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir('.', '/root')
)

CHECKPOINT_DIRS = dict(base="base_checkpoints", mid="mid_checkpoints", sft="chatsft_checkpoints", vl="vl_checkpoints")

def setup_env():
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

def has_checkpoint(phase):
    return bool(glob.glob(f"/data/{CHECKPOINT_DIRS[phase]}/*/model.pt"))

def ensure_tokenizer():
    if os.path.exists("/data/tokenizer.pkl"): return
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars=10000000", "--vocab_size=4096"], check=True)

def ensure_data():
    if glob.glob("/data/fineweb-edu/*.parquet"): return
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "--num-files=2"], check=True)

def ensure_base(batch_size):
    ensure_tokenizer()
    ensure_data()
    if has_checkpoint("base"): return
    subprocess.run(["python", "-m", "scripts.base_train", "--depth=8", "--num_iterations=20", f"--device_batch_size={batch_size}", f"--total_batch_size={batch_size*1024}", "--max_seq_len=1024", "--eval_every=-1", "--core_metric_every=-1", "--save_every=20"], check=True)

def ensure_mid(batch_size):
    ensure_base(batch_size)
    if has_checkpoint("mid"): return
    subprocess.run(["python", "-m", "scripts.mid_train", "--num_iterations=10", f"--device_batch_size={batch_size}", f"--total_batch_size={batch_size*1024}", "--max_seq_len=1024", "--eval_every=-1", "--save_every=10"], check=True)

def ensure_sft(batch_size):
    ensure_mid(batch_size)
    if has_checkpoint("sft"): return
    subprocess.run(["python", "-m", "scripts.chat_sft", "--num_iterations=417", f"--device_batch_size={batch_size}", "--target_examples_per_step=32", "--max_seq_len=1024", "--eval_every=-1", "--eval_steps=-1", "--save_every=10"], check=True)

def train_vl(use_images, num_steps, batch_size, target_examples):
    for p in glob.glob("/data/vl_checkpoints/*"): subprocess.run(["rm", "-rf", p])
    start = time.time()
    subprocess.run(["python", "-m", "scripts.vl_train", f"--num_steps={num_steps}", f"--device_batch_size={batch_size}", f"--target_examples_per_step={target_examples}", f"--use_images={use_images}", "--print_every=50"], check=True)
    return time.time() - start

def eval_vl(use_images, max_problems):
    args = ["python", "-m", "scripts.vl_eval", "--task-name=AOKVQA", f"--use-images={use_images}"]
    if max_problems: args.append(f"--max-problems={max_problems}")
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.split("\n"):
        if "accuracy:" in line.lower(): return float(line.split()[-1].replace("%", ""))
    return 0.0

@app.function(image=image, timeout=7200, gpu="L4", volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret"), wandb_secret])
def run_ablation(series_name: str = "test", num_steps: int = 100, batch_size: int = 4, target_examples: int = 32, eval_max: int = 50):
    import wandb, shutil
    setup_env()
    for f in os.listdir("/data"):
        p = f"/data/{f}"
        if os.path.isdir(p): shutil.rmtree(p)
        else: os.remove(p)
    print("Wiped volume, starting fresh...")
    ensure_sft(batch_size)
    volume.commit()
    
    print(f"\n{'='*60}\nEvaluating SFT model\n{'='*60}")
    subprocess.run(["python", "-m", "scripts.chat_eval", "-i", "sft", f"--max-problems={eval_max}"], check=True)
    return
    
    # wandb.init(project="nanochat-vl", name=f"{series_name}_ablation", config=dict(series=series_name, num_steps=num_steps, batch_size=batch_size, target_examples=target_examples, eval_max=eval_max))
    # 
    # results = []
    # for use_images in [1, 0]:
    #     config = "images" if use_images else "no_images"
    #     print(f"\n{'='*60}\nTraining: {config}\n{'='*60}")
    #     train_time = train_vl(use_images, num_steps, batch_size, target_examples)
    #     print(f"\n{'='*60}\nEvaluating: {config}\n{'='*60}")
    #     val_acc = eval_vl(use_images, eval_max)
    #     results.append(dict(config=config, val_acc=val_acc, train_time_min=train_time/60))
    #     print(f"{config}: val_acc={val_acc:.2f}%, train_time={train_time/60:.1f}min")
    # 
    # print(f"\n{'='*60}\nRESULTS: {series_name}\n{'='*60}")
    # print(f"{'config':<12} {'val_acc':>10} {'train_min':>10}")
    # for r in results: print(f"{r['config']:<12} {r['val_acc']:>9.2f}% {r['train_time_min']:>10.1f}")
    # 
    # diff = results[0]['val_acc'] - results[1]['val_acc']
    # print(f"\nDelta (images - no_images): {diff:+.2f}%")
    # verdict = "PASS: images help" if diff >= 3 else "FAIL: images don't help enough" if diff > 0 else "FAIL: images hurt"
    # print(verdict)
    # 
    # wandb.log(dict(images_acc=results[0]['val_acc'], no_images_acc=results[1]['val_acc'], delta=diff))
    # wandb.finish()
    # 
    # writer = csv.DictWriter(sys.stdout, fieldnames=["config", "val_acc", "train_time_min"])
    # writer.writeheader()
    # for r in results: writer.writerow(r)

@app.local_entrypoint()
def main(series: str = "test", steps: int = 2000, batch: int = 16, target: int = 32, eval_max: int = 50):
    run_ablation.remote(series, steps, batch, target, eval_max)
