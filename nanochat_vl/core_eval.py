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
    without = tpl.render(examples=examples, item=item, delim=delim, incl=False)
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
    return stack_sequences(seqs), [prefix_len - 1] * len(prompts), [len(s) - 1 for s in seqs]

def batch_sequences_schema(prompts, tokenizer):
    "For schema: each prompt is a context_option, find common suffix (continuation)"
    bos = tokenizer.get_bos_token_id()
    seqs = [tokenizer(p, prepend=bos) for p in prompts]
    suffix_len = find_common_length(seqs, 'right')
    return stack_sequences(seqs), [len(s) - suffix_len - 1 for s in seqs], [len(s) - 1 for s in seqs]

def batch_sequences_lm(prompts, tokenizer):
    "For LM: prompts is [without, with], score the continuation"
    bos = tokenizer.get_bos_token_id()
    without, with_ = prompts
    seq_without = tokenizer(without, prepend=bos)
    seq_with = tokenizer(with_, prepend=bos)
    prefix_len = find_common_length([seq_without, seq_with], 'left')
    return stack_sequences([seq_without, seq_with]), [prefix_len - 1], [len(seq_with) - 1]

def forward_model(model, tokens):
    "Forward pass, return per-token losses and argmax predictions"
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = model(tokens)
    targets, outputs = tokens[:, 1:], logits[:, :-1]
    losses = F.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), reduction='none').reshape(targets.shape)
    losses[:, -1] = float('nan')
    return losses, outputs.argmax(dim=-1)

@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    "Evaluate a single example, return True if correct"
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    delim = task_meta['continuation_delimiter']
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
    tokens = tokens[:, :max_seq_len].to(device)
    losses, preds = forward_model(model, tokens)
    if task_type in ('multiple_choice', 'schema'):
        mean_losses = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            e = min(e, max_seq_len - 1)
            mean_losses.append(losses[i, s:e].mean().item() if s < e else float('inf'))
        predicted = mean_losses.index(min(mean_losses))
        return predicted == item['gold']
    else:
        s, e = starts[0], min(ends[0], max_seq_len - 1)
        return preds[0, s:e].tolist() == tokens[1, s+1:e+1].tolist()

def evaluate_task(model, tokenizer, data, device, task_meta, max_examples=None):
    "Evaluate task, return accuracy"
    n = len(data) if max_examples is None else min(len(data), max_examples)
    correct = sum(evaluate_example(i, model, tokenizer, data, device, task_meta) for i in range(n))
    return correct / n
