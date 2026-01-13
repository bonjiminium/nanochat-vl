"Evaluate compression ratio of the tokenizer."

from nanochat_vl.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat_vl.dataset import parquets_iter_batched
from nanochat_vl.report import get_report

news_text = r"""(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid."""

korean_text = """한국어(韓國語)는 대한민국과 조선민주주의인민공화국의 공용어이다. 둘을 합쳐서 '남북한', '한반도'라고 부른다."""

code_text = """
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print([fibonacci(i) for i in range(20)])
"""

math_text = r"""The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ for $ax^2 + bx + c = 0$ where $a \neq 0$"""

science_text = """The mitochondria is the powerhouse of the cell. ATP (adenosine triphosphate) is synthesized through oxidative phosphorylation in the inner mitochondrial membrane."""

def get_text_samples():
    texts = dict(news=news_text, korean=korean_text, code=code_text, math=math_text, science=science_text)
    train_text, val_text = "", ""
    for i, batch in enumerate(parquets_iter_batched("train")):
        if i == 0:
            for doc in batch[:5]: train_text += doc
        elif i == 1:
            for doc in batch[:5]: val_text += doc
        else: break
    texts["train_sample"], texts["val_sample"] = train_text[:10000], val_text[:10000]
    return texts

def compression_ratio(tokenizer, text): return len(text.encode('utf-8')) / len(tokenizer.encode(text))

def colorize(val, ref, higher_is_better=True):
    green, red, reset = "\033[92m", "\033[91m", "\033[0m"
    better = val > ref if higher_is_better else val < ref
    color = green if better else red if val != ref else ""
    return f"{color}{val:.3f}{reset}"

def main():
    ours = get_tokenizer()
    gpt2 = RustBPETokenizer.from_pretrained("gpt2")
    gpt4 = RustBPETokenizer.from_pretrained("cl100k_base")
    texts = get_text_samples()
    results = {k: {} for k in ["gpt2", "gpt4", "ours"]}
    
    print(f"\n{'Text':<15} {'GPT-2':>10} {'GPT-4':>10} {'Ours':>10}")
    print("-" * 47)
    
    for name, text in texts.items():
        r_gpt2, r_gpt4, r_ours = compression_ratio(gpt2, text), compression_ratio(gpt4, text), compression_ratio(ours, text)
        nbytes = len(text.encode('utf-8'))
        results["gpt2"][name] = dict(bytes=nbytes, tokens=len(gpt2.encode(text)), ratio=r_gpt2)
        results["gpt4"][name] = dict(bytes=nbytes, tokens=len(gpt4.encode(text)), ratio=r_gpt4)
        results["ours"][name] = dict(bytes=nbytes, tokens=len(ours.encode(text)), ratio=r_ours)
        ref = max(r_gpt2, r_gpt4)
        print(f"{name:<15} {r_gpt2:>10.3f} {r_gpt4:>10.3f} {colorize(r_ours, ref):>20}")
    
    lines = []
    for baseline_name, baseline_key in [("GPT-2", "gpt2"), ("GPT-4", "gpt4")]:
        lines.append(f"### Comparison with {baseline_name}\n")
        lines.append(f"| Text | Bytes | {baseline_name} Tokens | {baseline_name} Ratio | Ours Tokens | Ours Ratio | Diff % |")
        lines.append("|------|-------|-------------|-------------|-------------|------------|--------|")
        for name in texts:
            b, o = results[baseline_key][name], results["ours"][name]
            diff = ((b['tokens'] - o['tokens']) / b['tokens']) * 100
            lines.append(f"| {name} | {b['bytes']} | {b['tokens']} | {b['ratio']:.2f} | {o['tokens']} | {o['ratio']:.2f} | {diff:+.1f}% |")
        lines.append("")
    get_report().log(section="Tokenizer evaluation", data=["\n".join(lines)])

if __name__ == "__main__": main()
