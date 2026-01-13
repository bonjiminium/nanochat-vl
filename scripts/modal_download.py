import modal, subprocess

app = modal.App("nanochat-vl-download")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests", "pyarrow", "rustbpe", "tiktoken", "torch")
    .add_local_dir('.', '/root')
)

@app.function(image=image, timeout=3600)
def download(n_shards: int = 8):
    subprocess.run(["python", "-m", "nanochat_vl.dataset", "-n", str(n_shards)], check=True)

@app.local_entrypoint()
def main(n: int = 8): download.remote(n)
