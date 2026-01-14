"Evaluate a Chat model on benchmarks."

import argparse
from functools import partial
import torch
from nanochat_vl.checkpoint_manager import load_model
from nanochat_vl.tokenizer import get_tokenizer
from tasks.arc import ARC

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    "Run categorical (multiple choice) evaluation."
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)
    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad(): logits = model(prompt_tensor)

        for idx, conv in enumerate(conversations):
            letters = conv['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    enc = tokenizer.encode(letter)
                    assert len(enc) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = enc[0]
                letter_ids.append(letter_to_id_cache[letter])
            focus_logits = logits[idx, answer_positions[idx], letter_ids]
            predicted = letters[focus_logits.argmax(dim=-1).item()]
            num_passed += int(task_object.evaluate(conv, predicted))
            total += 1

    acc = num_passed / total
    print(f"Final: {num_passed}/{total} ({100*acc:.2f}%)")
    return acc

def run_chat_eval(task_name, model, tokenizer, batch_size=8, max_problems=None):
    task_module = {
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
    }[task_name]
    task_object = task_module()
    if task_object.eval_type == 'categorical': return run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems)
    raise ValueError(f"Unsupported eval type: {task_object.eval_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source: base|mid|sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name (default=all)")
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-x', '--max-problems', type=int, default=None)
    args = parser.parse_args()

    model, _, meta = load_model(args.source, "cuda", phase="eval")
    model = torch.compile(model)
    tokenizer = get_tokenizer()

    all_tasks = ['ARC-Easy', 'ARC-Challenge']
    baseline_accuracies = {'ARC-Easy': 0.25, 'ARC-Challenge': 0.25}
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    results = {}
    for task_name in task_names:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            acc = run_chat_eval(task_name, model, tokenizer, batch_size=args.batch_size, max_problems=args.max_problems)
            results[task_name] = acc
            print(f"{task_name} accuracy: {100*acc:.2f}%")

    from nanochat_vl.report import get_report
    chatcore_metric_dict = {}
    if all(t in results for t in all_tasks):
        centered_mean = sum((results[t] - baseline_accuracies[t]) / (1 - baseline_accuracies[t]) for t in results) / len(results)
        chatcore_metric_dict = {"ChatCORE metric": centered_mean}
    get_report().log(section=f"Chat evaluation {args.source}", data=[vars(args), results, chatcore_metric_dict])
