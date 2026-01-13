"Evaluate bpb on train/val and sample from the model."
import argparse, torch
from nanochat_vl.checkpoint_manager import load_model
from nanochat_vl.dataloader import data_loader
from nanochat_vl.tokenizer import get_tokenizer, get_token_bytes
from nanochat_vl.loss_eval import evaluate_bpb
from nanochat_vl.report import get_report
from nanochat_vl.engine import Engine

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=-1)
parser.add_argument('--eval_tokens', type=int, default=10*524288)
parser.add_argument('--device_batch_size', type=int, default=16)
args = parser.parse_args()

step = args.step if args.step >= 0 else None
model, tokenizer, meta = load_model(source='base', device='cuda', phase='eval', step=step)
model.eval()
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device='cuda')
seq_len = model.config.seq_len
steps = args.eval_tokens // (args.device_batch_size * seq_len)

bpb_results = {}
for split in ['train', 'val']:
    loader = data_loader(args.device_batch_size, seq_len, split, device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print(f"{split} bpb: {bpb:.4f}")
    bpb_results[split] = bpb

prompts = ["The capital of France is", "The chemical symbol of gold is", "If yesterday was Friday, then tomorrow will be", "The opposite of hot is", "The planets of the solar system are:", "My favorite color is", "If 5*x + 3 = 13, then x is"]

engine = Engine(model, tokenizer)
samples = []
for prompt in prompts:
    gen = engine.generate(prompt, max_tokens=32, temperature=0.0)
    text = f"{prompt}{gen}"
    print(text)
    samples.append(text)

sample_dict = {f"sample {i}": s for i, s in enumerate(samples)}
get_report().log(section="Base model loss", data=[{"train_bpb": bpb_results["train"], "val_bpb": bpb_results["val"]}, sample_dict])
