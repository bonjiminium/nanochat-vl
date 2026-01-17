"Functions for evaluating the CORE metric, as described in the DCLM paper."
import random
from jinja2 import Template
import torch, torch.nn.functional as F

def render_prompts_mc(item, delim, examples=None):
    "Multiple choice: same question, different answer choices"
    tpl = Template("""{%- for ex in examples -%}{{ ex.query }}{{ delim }}{{ ex.choices[ex.gold] }}\n\n{% endfor -%}{{ item.query }}{{ delim }}{{ choice }}""".strip())
    examples = examples or []
    return [tpl.render(examples=examples, item=item, delim=delim, choice=c) for c in item['choices']]

def render_prompts_schema(item, delim, examples=None):
    "Schema: different contexts, same continuation"
    tpl = Template("""{%- for ex in examples -%}{{ ex.context_options[ex.gold] }}{{ delim }}{{ ex.continuation }}\n\n{% endfor -%}{{ ctx }}{{ delim }}{{ item.continuation }}""".strip())
    examples = examples or []
    return [tpl.render(examples=examples, item=item, delim=delim, ctx=c) for c in item['context_options']]

def render_prompts_lm(item, delim, examples=None):
    "Language modeling: context -> continuation"
    tpl = Template("""{%- for ex in examples -%}{{ ex.context | trim }}{{ delim }}{{ ex.continuation }}\n\n{% endfor -%}{{ item.context | trim }}{{ delim }}{% if incl %}{{ item.continuation }}{% endif %}""".strip())
    examples = examples or []
    without = tpl.render(examples=examples, item=item, delim=delim, incl=False).strip()
    with_ = tpl.render(examples=examples, item=item, delim=delim, incl=True)
    return [without, with_]

def find_common_length(seqs, direction='left'):
    "Find length of common prefix (left) or suffix (right)"
    min_len = min(len(s) for s in seqs)
    if direction == 'right': seqs = [s[::-1] for s in seqs]
    for i in range(min_len):
        if len(set(s[i] for s in seqs)) > 1: return i
    return min_len

def stack_sequences(seqs, pad_value=0):
    "Pad sequences to same length and stack"
    max_len = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_value] * (max_len - len(s)) for s in seqs], dtype=torch.long)

def batch_sequences_mc(prompts, tokenizer):
    "For MC: all prompts share prefix, score the different suffixes"
    bos = tokenizer.get_bos_token_id()
    seqs = [tokenizer(p, prepend=bos) for p in prompts]
    prefix_len = find_common_length(seqs, 'left')
    return stack_sequences(seqs), [prefix_len] * len(prompts), [len(s) for s in seqs]

def batch_sequences_schema(prompts, tokenizer):
    "For schema: each prompt is a context_option, find common suffix (continuation)"
    bos = tokenizer.get_bos_token_id()
    seqs = [tokenizer(p, prepend=bos) for p in prompts]
    suffix_len = find_common_length(seqs, 'right')
    return stack_sequences(seqs), [len(s) - suffix_len for s in seqs], [len(s) for s in seqs]

def batch_sequences_lm(prompts, tokenizer):
    "For LM: prompts is [without, with], score the continuation (batch size 1)"
    bos = tokenizer.get_bos_token_id()
    without, with_ = prompts
    seq_without = tokenizer(without, prepend=bos)
    seq_with = tokenizer(with_, prepend=bos)
    start_idx, end_idx = len(seq_without), len(seq_with)
    assert start_idx < end_idx and seq_without == seq_with[:start_idx]
    return stack_sequences([seq_with]), [start_idx], [end_idx]

def forward_model(model, tokens):
    "Forward pass, return per-token losses and argmax predictions (shape B,T like Karpathy)"
    B, T = tokens.shape
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = model(tokens)
    targets = torch.roll(tokens, shifts=-1, dims=1)
    losses = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T), reduction='none').view(B, T)
    losses[:, -1] = float('nan')
    return losses, logits.argmax(dim=-1)

@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    "Evaluate a single example, return True if correct"
    item = data[idx]
    task_type, num_fewshot, delim = task_meta['task_type'], task_meta['num_fewshot'], task_meta['continuation_delimiter']
    examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        examples = rng.sample([d for i, d in enumerate(data) if i != idx], min(num_fewshot, len(data) - 1))
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, delim, examples)
        tokens, starts, ends = batch_sequences_mc(prompts, tokenizer)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, delim, examples)
        tokens, starts, ends = batch_sequences_schema(prompts, tokenizer)
    else:
        prompts = render_prompts_lm(item, delim, examples)
        tokens, starts, ends = batch_sequences_lm(prompts, tokenizer)
    max_seq_len = model.config.seq_len
    new_tokens, new_starts, new_ends = [], [], []
    for i in range(tokens.shape[0]):
        t, s, e = tokens[i].tolist()[:ends[i]], starts[i], ends[i]
        if len(t) > max_seq_len:
            crop = len(t) - max_seq_len
            t, s, e = t[-max_seq_len:], s - crop, e - crop
        new_tokens.append(t)
        new_starts.append(s)
        new_ends.append(e)
    if any(s < 0 or e < 0 for s, e in zip(new_starts, new_ends)): return False
    tokens, starts, ends = stack_sequences(new_tokens).to(device), new_starts, new_ends
    losses, preds = forward_model(model, tokens)
    if task_type == 'language_modeling':
        si, ei = starts[0], min(ends[0], max_seq_len)
        return torch.all(preds[0, si-1:ei-1] == tokens[0, si:ei]).item()
    elif task_type in ('multiple_choice', 'schema'):
        mean_losses = [losses[i, si-1:ei-1].mean().item() for i, (si, ei) in enumerate(zip(starts, ends))]
        return mean_losses.index(min(mean_losses)) == item['gold']
    else: raise ValueError(f"Unsupported task type: {task_type}")

def evaluate_task(model, tokenizer, data, device, task_meta, max_examples=None):
    "Evaluate task, return accuracy"
    n = len(data) if max_examples is None else min(len(data), max_examples)
    return sum(evaluate_example(i, model, tokenizer, data, device, task_meta) for i in range(n)) / n
