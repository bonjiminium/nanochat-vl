"Evaluate the VLM on vision benchmarks."

import os, argparse, copy, logging
os.environ["TORCH_LOGS"] = "-dynamo"
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*IndexPutBackward0.*")
import torch
from functools import partial
from nanochat_vl.vlm import VLM
from nanochat_vl.common import get_base_dir
from pathlib import Path
from nanochat_vl.checkpoint_manager import load_model, load_checkpoint, find_last_checkpoint_dir, find_last_step
from nanochat_vl.vl_dataloader import process_image
from tasks.scienceqa import ScienceQA
from tasks.aokvqa import AOKVQA

def render_for_completion_vl(tokenizer, conversation, img_token_id, num_patches):
    "Render conversation for completion with image placeholders expanded to image tokens."
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant", "Last message must be from Assistant"
    messages.pop()
    ids = [tokenizer.get_bos_token_id()]
    def encode_with_images(content):
        parts = content.split("<|image|>")
        for i, part in enumerate(parts):
            ids.extend(tokenizer.encode(part))
            if i < len(parts) - 1: ids.extend([img_token_id] * num_patches)
    for msg in messages:
        if msg["role"] == "user":
            ids.append(tokenizer.encode_special("<|user_start|>"))
            encode_with_images(msg["content"])
            ids.append(tokenizer.encode_special("<|user_end|>"))
    ids.append(tokenizer.encode_special("<|assistant_start|>"))
    return ids

def run_categorical_eval(task_object, tokenizer, vlm, batch_size, img_size, patch_size, img_token_id, max_problems=None):
    device = next(vlm.parameters()).device
    bos = tokenizer.get_bos_token_id()
    num_patches = (img_size // patch_size) ** 2
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)
    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        all_imgs = []
        for conv in conversations:
            imgs = conv.get("images", [])
            all_imgs.extend([process_image(img, img_size) for img in imgs])
        imgs_tensor = torch.stack(all_imgs).to(device) if all_imgs else torch.empty(0, 3, img_size, img_size, device=device)
        prompt_ids = [render_for_completion_vl(tokenizer, conv, img_token_id, num_patches) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad(): logits = vlm(imgs_tensor, prompt_tensor, img_token_id=img_token_id)

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
            correct = task_object.evaluate(conv, predicted)
            expected = conv["messages"][-1]["content"]
            print(f"[{i0+idx:3d}] pred={predicted} exp={expected} {'✓' if correct else '✗'}")
            num_passed += int(correct)
            total += 1

        print(f"Running: {num_passed}/{total} ({100*num_passed/total:.1f}%)")

    print()
    acc = num_passed / total
    print(f"Final: {num_passed}/{total} ({100*acc:.2f}%)")
    return acc

def run_vl_eval(task_name, vlm, tokenizer, batch_size, img_size, patch_size, img_token_id, max_problems=None, use_images=True):
    task_module = {
        'ScienceQA': partial(ScienceQA, split="test", only_images=True),
        'AOKVQA': partial(AOKVQA, split="validation", use_images=use_images),
    }[task_name]
    task_object = task_module()
    if task_object.eval_type == 'categorical': return run_categorical_eval(task_object, tokenizer, vlm, batch_size, img_size, patch_size, img_token_id, max_problems)
    raise ValueError(f"Unsupported eval type: {task_object.eval_type}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-a', '--task_name', type=str, default='AOKVQA')
    p.add_argument('-b', '--batch_size', type=int, default=8)
    p.add_argument('-x', '--max_problems', type=int, default=None)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--use_images', type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path(get_base_dir())
    gpt, tokenizer, _ = load_model("sft", device, phase="eval")
    img_token_id = tokenizer.encode_special("<image>")
    vl_ckpt_dir = find_last_checkpoint_dir(base_dir / "vl_checkpoints")
    if vl_ckpt_dir:
        step = find_last_step(vl_ckpt_dir)
        print(f"Loading VLM checkpoint from {vl_ckpt_dir} step {step}")
        model_data, _, meta = load_checkpoint(vl_ckpt_dir, step, device)
        vlm_cfg = meta.get("vlm_config", {})
        vlm = VLM(gpt, vlm_cfg.get("img_size", args.img_size), vlm_cfg.get("patch_size", args.patch_size), vlm_cfg.get("vision_dim", 256)).to(device)
        vlm.load_state_dict(model_data)
    else:
        print("No VLM checkpoint found, using random init")
        vlm = VLM(gpt, img_size=args.img_size, patch_size=args.patch_size).to(device)
    vlm = torch.compile(vlm)
    vlm.eval()

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        acc = run_vl_eval(args.task_name, vlm, tokenizer, args.batch_size, args.img_size, args.patch_size, img_token_id, args.max_problems, use_images=bool(args.use_images))
        print(f"{args.task_name} accuracy: {100*acc:.2f}%")
