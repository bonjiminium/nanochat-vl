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
    ensure_tokenizer()
    import torch
    from nanochat_vl.tokenizer import get_tokenizer
    from nanochat_vl.vl_dataloader import vl_data_generator
    from nanochat_vl.vlm import VLM
    from nanochat_vl.gpt import GPT, GPTConfig
    from tasks.scienceqa import ScienceQA
    
    tok = get_tokenizer()
    img_token_id = tok.encode_special("<image>")
    img_size, patch_size = 64, 8
    num_patches = (img_size // patch_size) ** 2
    
    print("="*60)
    print("EMBEDDING INJECTION VERIFICATION")
    print("="*60)
    
    print(f"\n[1] Config: img_size={img_size}, patch_size={patch_size}, num_patches={num_patches}, img_token_id={img_token_id}")
    
    ds = ScienceQA(split="train")
    gen = vl_data_generator(ds, tok, batch_size=2, img_size=img_size, max_seq_len=256, num_patches=num_patches, device="cuda")
    imgs, inputs, targets = next(gen)
    
    print(f"\n[2] Dataloader output:")
    print(f"    imgs.shape: {imgs.shape} (expected: [N, 3, {img_size}, {img_size}])")
    print(f"    inputs.shape: {inputs.shape}")
    print(f"    targets.shape: {targets.shape}")
    
    mask = (inputs == img_token_id)
    num_img_tokens = mask.sum().item()
    num_images = imgs.shape[0]
    expected_tokens = num_images * num_patches
    print(f"\n[3] Token count check:")
    print(f"    Images in batch: {num_images}")
    print(f"    Image tokens in inputs: {num_img_tokens}")
    print(f"    Expected (images * patches): {expected_tokens}")
    print(f"    MATCH: {num_img_tokens == expected_tokens}")
    
    print(f"\n[4] Token positions per sequence:")
    for b in range(inputs.shape[0]):
        seq_mask = mask[b]
        positions = seq_mask.nonzero(as_tuple=True)[0].tolist()
        if positions: print(f"    seq {b}: {len(positions)} tokens at positions {positions[0]}..{positions[-1]}")
        else: print(f"    seq {b}: no image tokens")
    
    cfg = GPTConfig(seq_len=256, vocab_size=4096, n_layer=2, n_head=2, n_kv_head=2, n_embd=128)
    gpt = GPT(cfg).cuda().bfloat16()
    gpt.init_weights()
    vlm = VLM(gpt, vision_dim=256, img_size=img_size, patch_size=patch_size, vit_layers=2).cuda().bfloat16()
    
    print(f"\n[5] Embedding shapes in forward pass:")
    tok_emb = torch.nn.functional.rms_norm(gpt.transformer.wte(inputs), (gpt.config.n_embd,))
    img_emb = vlm.proj(vlm.vit(imgs.to(tok_emb.dtype))).to(tok_emb.dtype)
    print(f"    tok_emb before injection: {tok_emb.shape}")
    print(f"    img_emb from ViT+proj: {img_emb.shape} -> flattened: {img_emb.view(-1, img_emb.size(-1)).shape}")
    
    print(f"\n[6] Embedding values at image positions (before vs after):")
    tok_emb_before = tok_emb.clone()
    img_emb_flat = img_emb.view(-1, img_emb.size(-1))
    tok_emb[mask] = img_emb_flat[:num_img_tokens]
    diff = (tok_emb[mask] - tok_emb_before[mask]).abs().mean().item()
    print(f"    Mean abs diff at image positions: {diff:.4f}")
    print(f"    First image emb (5 vals): {img_emb_flat[0, :5].tolist()}")
    print(f"    tok_emb at first img pos (5 vals): {tok_emb[mask][0, :5].tolist()}")
    print(f"    INJECTED CORRECTLY: {torch.allclose(tok_emb[mask], img_emb_flat[:num_img_tokens])}")
    
    print(f"\n[7] Forward + backward:")
    loss = vlm(imgs, inputs, targets, img_token_id=img_token_id)
    print(f"    loss: {loss.item():.4f}")
    loss.backward()
    vit_grad = vlm.vit.patch_embed.proj.weight.grad
    proj_grad = vlm.proj.proj.weight.grad
    print(f"    ViT patch_embed grad norm: {vit_grad.norm().item():.6f}")
    print(f"    Projector grad norm: {proj_grad.norm().item():.6f}")
    print(f"    GRADIENTS FLOW: {vit_grad.norm().item() > 0 and proj_grad.norm().item() > 0}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

@app.local_entrypoint()
def main(test: str = "", run: str = "dummy"):
    if test == "base": return test_base.remote()
    if test == "mid": return test_mid.remote()
    if test == "sft": return test_sft.remote()
    if test == "vl": return test_vl.remote()
    if test == "temp": return test_temp.remote()
    from nanochat_vl.report import get_git_info, get_bloat_info
    speedrun.remote(run=run, git_info=get_git_info(), bloat_info=get_bloat_info())
