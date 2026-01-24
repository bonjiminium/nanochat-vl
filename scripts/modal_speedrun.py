import modal, subprocess, os, glob

app = modal.App("nanochat-vl-speedrun")
volume = modal.Volume.from_name("nanochat-vl-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy", "psutil", "pyyaml", "jinja2", "wandb", "datasets", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir('.', '/root')
)

CHECKPOINT_DIRS = dict(base="base_checkpoints", mid="mid_checkpoints", sft="chatsft_checkpoints", vl="vl_checkpoints")

TOK_TEST = dict(max_chars=10_000_000, vocab_size=4096)
TOK_REAL = dict(max_chars=2_000_000_000, vocab_size=65536)

BASE_TEST = dict(depth=2, aspect_ratio=64, head_dim=64, max_seq_len=1024, vocab_size=4096, device_batch_size=4, total_batch_size=4096, num_iterations=20, warmup_ratio=0.1, warmdown_ratio=0.2, eval_every=5, eval_tokens=4096, core_metric_every=-1)
BASE_REAL = dict(depth=16, vocab_size=65536, device_batch_size=32)

MID_TEST = dict(num_iterations=10, device_batch_size=4, total_batch_size=4096, max_seq_len=1024, eval_every=5, eval_tokens=4096)
MID_REAL = dict(num_iterations=2187, device_batch_size=32, total_batch_size=524288, eval_every=150)

SFT_TEST = dict(num_iterations=10, device_batch_size=4, target_examples_per_step=32, max_seq_len=1024, eval_every=5, eval_steps=4)
SFT_REAL = dict(num_iterations=417, device_batch_size=32, target_examples_per_step=32, max_seq_len=1024)

VL_TEST = dict(num_steps=20, device_batch_size=2, target_examples_per_step=8, print_every=5)
VL_REAL = dict(num_steps=2000, device_batch_size=32, target_examples_per_step=32, max_seq_len=1024, print_every=50)

def setup_env():
    os.environ["NANOCHAT_VL_BASE_DIR"] = "/data"
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

def run(cmd): subprocess.run(cmd, check=True)
def run_script(script, cfg): run(["python", "-u", "-m", script] + [f"--{k}={v}" for k,v in cfg.items()])

def has_checkpoint(stage):
    from nanochat_vl.common import get_base_dir
    d = os.path.join(get_base_dir(), CHECKPOINT_DIRS[stage])
    return bool(glob.glob(os.path.join(d, "d*", "model_*.pt")))

def ensure_tokenizer(test=True):
    from nanochat_vl.common import get_base_dir
    from pathlib import Path
    tok_path = Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl"
    expected_vocab = TOK_TEST["vocab_size"] if test else TOK_REAL["vocab_size"]
    if tok_path.exists():
        import pickle
        with open(tok_path, "rb") as f: enc = pickle.load(f)
        if enc.n_vocab == expected_vocab: return
        print(f"Tokenizer vocab mismatch: found {enc.n_vocab}, expected {expected_vocab}. Retraining...")
    run(["python", "-m", "nanochat_vl.dataset", "-n", "2" if test else "240"])
    run_script("scripts.tok_train", TOK_TEST if test else TOK_REAL)

def ensure_base(test=True):
    ensure_tokenizer(test)
    if has_checkpoint("base"): return
    run_script("scripts.base_train", BASE_TEST if test else BASE_REAL)

def ensure_mid(test=True):
    ensure_base(test)
    if has_checkpoint("mid"): return
    run_script("scripts.mid_train", MID_TEST if test else MID_REAL)

def ensure_sft(test=True):
    ensure_mid(test)
    if has_checkpoint("sft"): return
    run_script("scripts.chat_sft", SFT_TEST if test else SFT_REAL)

def ensure_vl(test=True):
    ensure_sft(test)
    if has_checkpoint("vl"): return
    run_script("scripts.vl_train", VL_TEST if test else VL_REAL)

def gpu(test): return "L4" if test else "H100"
def timeout(test): return 600 if test else 14400

@app.function(image=image, gpu="L4", timeout=600, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def base_test(eval_only: bool = False):
    setup_env()
    if not eval_only:
        ensure_tokenizer(test=True)
        run_script("scripts.base_train", BASE_TEST)
    if eval_only:
        run_script("scripts.base_loss", dict(eval_tokens=4096, device_batch_size=4))
        run_script("scripts.base_eval", dict(max_problems=50, batch_size=4))
    volume.commit()

@app.function(image=image, gpu="H100", timeout=14400, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def base_real(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, eval_only: bool = False):
    setup_env()
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    if not eval_only:
        ensure_tokenizer(test=False)
        run_script("scripts.tok_eval", {})
        run_script("scripts.base_train", {**BASE_REAL, "run": run})
    run_script("scripts.base_loss", dict(eval_tokens=10485760, device_batch_size=32))
    run_script("scripts.base_eval", dict(max_problems=500, batch_size=32))
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())
    volume.commit()

@app.function(image=image, gpu="L4", timeout=600, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def mid_test(eval_only: bool = False):
    setup_env()
    if not eval_only: 
        ensure_base(test=True)
        run_script("scripts.mid_train", MID_TEST)
    if eval_only: run_script("scripts.chat_eval", dict(source="mid", max_problems=50))
    volume.commit()

@app.function(image=image, gpu="H100", timeout=14400, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def mid_real(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, eval_only: bool = False):
    setup_env()
    ensure_tokenizer(test=False)
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    if not eval_only:
        ensure_base(test=False)
        run_script("scripts.mid_train", {**MID_REAL, "run": run})
    run_script("scripts.chat_eval", dict(source="mid", max_problems=500))
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())
    volume.commit()

@app.function(image=image, gpu="L4", timeout=600, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def sft_test(eval_only: bool = False):
    setup_env()
    if not eval_only:
        ensure_mid(test=True)
        run_script("scripts.chat_sft", SFT_TEST)
    if eval_only: run_script("scripts.chat_eval", dict(source="sft", max_problems=50))
    volume.commit()

@app.function(image=image, gpu="H100", timeout=14400, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def sft_real(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, eval_only: bool = False):
    setup_env()
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    if not eval_only:
        ensure_mid(test=False)
        run_script("scripts.chat_sft", {**SFT_REAL, "run": run})
    run_script("scripts.chat_eval", dict(source="sft", max_problems=500))
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())
    volume.commit()

VIT_TEST = {"epochs": 1, "batch-size": 8, "log-every": 5, "img-size": 64}
VIT_REAL = {"epochs": 10, "batch-size": 32, "log-every": 50, "img-size": 64}

@app.function(image=image, gpu="L4", timeout=600, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def test_temp():
    setup_env()
    run_script("scripts.vit_pretrain", VIT_TEST)
    volume.commit()

@app.function(image=image, gpu="L4", timeout=600, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def vl_test(eval_only: bool = False):
    setup_env()
    if not eval_only:
        ensure_sft(test=True)
        run_script("scripts.vl_train", VL_TEST)
    if eval_only: run_script("scripts.vl_eval", dict(task_name="AOKVQA", max_problems=50, use_images=1))
    volume.commit()

@app.function(image=image, gpu="H100", timeout=14400, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def vl_real(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, eval_only: bool = False):
    setup_env()
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    if not eval_only:
        ensure_sft(test=False)
        run_script("scripts.vl_train", {**VL_REAL, "run": run})
    run_script("scripts.vl_eval", dict(task_name="AOKVQA", max_problems=500, use_images=1))
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())
    volume.commit()

@app.function(image=image, gpu="L4", timeout=1800, volumes={"/data": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])
def speedrun_test(eval_only: bool = False):
    setup_env()
    if not eval_only:
        ensure_tokenizer(test=True)
        run_script("scripts.base_train", BASE_TEST)
        run_script("scripts.mid_train", MID_TEST)
        run_script("scripts.chat_sft", SFT_TEST)
        run_script("scripts.vl_train", VL_TEST)
    if eval_only:
        run_script("scripts.base_loss", dict(eval_tokens=4096, device_batch_size=4))
        run_script("scripts.base_eval", dict(max_problems=50, batch_size=4))
        run_script("scripts.chat_eval", dict(source="mid", max_problems=50))
        run_script("scripts.chat_eval", dict(source="sft", max_problems=50))
        run_script("scripts.vl_eval", dict(task_name="AOKVQA", max_problems=50, use_images=1))
    volume.commit()

@app.function(image=image, gpu="H100", timeout=28800, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")])
def speedrun_real(run: str = "dummy", git_info: dict = None, bloat_info: dict = None, eval_only: bool = False):
    setup_env()
    from nanochat_vl.common import get_base_dir
    from nanochat_vl.report import get_report, get_gpu_info, get_system_info, estimate_cost, get_dep_count
    report = get_report()
    report.reset(git_info or {}, bloat_info or {}, get_gpu_info(), get_system_info(), estimate_cost(get_gpu_info()), get_dep_count())
    if not eval_only:
        ensure_tokenizer(test=False)
        run_script("scripts.tok_eval", {})
        run_script("scripts.base_train", {**BASE_REAL, "run": run})
        run_script("scripts.mid_train", {**MID_REAL, "run": run})
        run_script("scripts.chat_sft", {**SFT_REAL, "run": run})
        run_script("scripts.vl_train", {**VL_REAL, "run": run})
    run_script("scripts.base_loss", dict(eval_tokens=10485760, device_batch_size=32))
    run_script("scripts.base_eval", dict(max_problems=500, batch_size=32))
    run_script("scripts.chat_eval", dict(source="mid", max_problems=500))
    run_script("scripts.chat_eval", dict(source="sft", max_problems=500))
    run_script("scripts.vl_eval", dict(task_name="AOKVQA", max_problems=500, use_images=1))
    report.generate()
    print(open(os.path.join(get_base_dir(), "report", "report.md")).read())
    volume.commit()

@app.local_entrypoint()
def main(stage: str = "speedrun", test: bool = False, run: str = "dummy", eval_only: bool = False):
    funcs = dict(base=(base_test, base_real), mid=(mid_test, mid_real), sft=(sft_test, sft_real), vl=(vl_test, vl_real), speedrun=(speedrun_test, speedrun_real))
    if stage not in funcs: raise ValueError(f"Unknown stage: {stage}. Choose from {list(funcs.keys())}")
    test_fn, real_fn = funcs[stage]
    if test: test_fn.remote(eval_only=eval_only)
    else:
        from nanochat_vl.report import get_git_info, get_bloat_info
        real_fn.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info(), eval_only=eval_only)
