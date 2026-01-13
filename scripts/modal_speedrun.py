import modal, subprocess

app = modal.App("nanochat-vl-speedrun")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch", "numpy")
    .add_local_dir('.', '/root')
)

@app.function(image=image, timeout=3600)
def speedrun(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536):
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", str(n_shards)], check=True)
    subprocess.run(["python", "-m", "scripts.tok_train", "--max_chars", str(max_chars), "--vocab_size", str(vocab_size)], check=True)
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)

@app.local_entrypoint()
def main(n_shards: int = 8, max_chars: int = 2_000_000_000, vocab_size: int = 65536): speedrun.remote(n_shards, max_chars, vocab_size)
