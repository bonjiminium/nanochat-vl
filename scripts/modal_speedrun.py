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

@app.local_entrypoint()
def main(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536):
    from nanochat_vl.report import get_git_info, get_bloat_info
    git_info, bloat_info = get_git_info(), get_bloat_info()
    speedrun.remote(n_shards, max_chars, vocab_size, git_info, bloat_info)
