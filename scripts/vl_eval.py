"Evaluate the VLM on vision benchmarks."

import argparse
import torch
from functools import partial
from nanochat_vl.vlm import VLM
from nanochat_vl.common import get_base_dir
from pathlib import Path
from nanochat_vl.checkpoint_manager import load_model, load_checkpoint, find_last_checkpoint_dir, find_last_step
from nanochat_vl.vl_dataloader import process_image
from tasks.scienceqa import ScienceQA

def run_categorical_eval(task_object, tokenizer, vlm, batch_size, img_size, max_problems=None):
    device = next(vlm.parameters()).device
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)
    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        imgs = torch.stack([process_image(conv["image"], img_size) for conv in conversations]).to(device)
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad(): logits = vlm(imgs, prompt_tensor)

        for idx, conv in enumerate(conversations):
            letters = conv['letters']
            for letter in letters:
                if letter not in letter_to_id_cache:
                    enc = tokenizer.encode(letter)
                    assert len(enc) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = enc[0]
            letter_ids = [letter_to_id_cache[l] for l in letters]
            focus_logits = logits[idx, answer_positions[idx], letter_ids]
            predicted = letters[focus_logits.argmax(dim=-1).item()]
            num_passed += int(task_object.evaluate(conv, predicted))
            total += 1

        print(f"\r{num_passed}/{total} ({100*num_passed/total:.1f}%)", end='', flush=True)

    print()
    acc = num_passed / total
    print(f"Final: {num_passed}/{total} ({100*acc:.2f}%)")
    return acc

def run_vl_eval(task_name, vlm, tokenizer, batch_size, img_size, max_problems=None):
    task_module = {'ScienceQA': partial(ScienceQA, split="test", only_images=True)}[task_name]
    task_object = task_module()
    if task_object.eval_type == 'categorical': return run_categorical_eval(task_object, tokenizer, vlm, batch_size, img_size, max_problems)
    raise ValueError(f"Unsupported eval type: {task_object.eval_type}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-a', '--task-name', type=str, default='ScienceQA')
    p.add_argument('-b', '--batch-size', type=int, default=8)
    p.add_argument('-x', '--max-problems', type=int, default=None)
    p.add_argument('--img-size', type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path(get_base_dir())
    gpt, tokenizer, _ = load_model("sft", device, phase="eval")
    vl_ckpt_dir = find_last_checkpoint_dir(base_dir / "vl_checkpoints")
    if vl_ckpt_dir:
        step = find_last_step(vl_ckpt_dir)
        print(f"Loading VLM checkpoint from {vl_ckpt_dir} step {step}")
        model_data, _, meta = load_checkpoint(vl_ckpt_dir, step, device)
        vlm_cfg = meta.get("vlm_config", {})
        vlm = VLM(gpt, vlm_cfg.get("img_size", args.img_size), vlm_cfg.get("patch_size", 8), vlm_cfg.get("vision_dim", 256)).to(device)
        vlm.load_state_dict(model_data)
    else:
        print("No VLM checkpoint found, using random init")
        vlm = VLM(gpt, img_size=args.img_size).to(device)
    vlm = torch.compile(vlm)
    vlm.eval()

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        acc = run_vl_eval(args.task_name, vlm, tokenizer, args.batch_size, args.img_size, args.max_problems)
        print(f"{args.task_name} accuracy: {100*acc:.2f}%")

    from nanochat_vl.report import get_report
    get_report().log(section="VL Evaluation", data=[
        {"task": args.task_name},
        {"num_problems": args.max_problems or len(ScienceQA(split="test", only_images=True))},
        {"accuracy": f"{100*acc:.2f}%"},
    ])
