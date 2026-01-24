"Generate matplotlib plot dataset for ViT pretraining."

import random, numpy as np, matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

FIGSIZES = [(3,3), (4,3), (3,4), (4,4), (5,4), (4,5), (5,5)]

def generate_plot(color=0, x_offset=0, period=0.5, y_exponent=-5, amplitude=0.5, y_offset=0, x_center=0, x_span=2.0, figsize=(4,4)):
    "Generate a matplotlib plot with one sine wave, return PIL Image and labels"
    fig, ax = plt.subplots(figsize=figsize)
    x_start, x_end = x_center - x_span/2, x_center + x_span/2
    x = np.linspace(x_start, x_end, 200)
    y = (amplitude * np.sin(2 * np.pi * (x - x_offset * period) / period) + y_offset) * (10 ** y_exponent)
    ax.plot(x, y, color=f'C{color}')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    fig.canvas.draw()
    actual_exp = ax.yaxis.get_offset_text().get_text()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), dict(color=color, y_exponent=actual_exp)

def gen_examples(n, seed_offset=0):
    for i in range(n):
        rng = random.Random(seed_offset + i)
        factors = dict(
            color=rng.randint(0, 9),
            x_offset=rng.randint(0, 99) / 100,
            period=0.5 + rng.randint(0, 19) * 0.125,
            y_exponent=-8 + rng.randint(0, 6),
            amplitude=0.3 + rng.randint(0, 12) * 0.05,
            y_offset=-0.5 + rng.randint(0, 20) * 0.05,
            x_center=-1 + rng.randint(0, 20) * 0.1,
            x_span=0.3 + rng.randint(0, 22) * 0.1,
            figsize=rng.choice(FIGSIZES),
        )
        img, labels = generate_plot(**factors)
        yield dict(image=img, **labels)

features = Features({"image": HFImage(), "color": Value("int32"), "y_exponent": Value("string")})

def make_dataset_dict(n_train=100_000, n_val=1_000, n_test=1_000):
    return DatasetDict({
        "train": Dataset.from_generator(lambda: gen_examples(n_train, 0), features=features),
        "validation": Dataset.from_generator(lambda: gen_examples(n_val, 1_000_000_000), features=features),
        "test": Dataset.from_generator(lambda: gen_examples(n_test, 2_000_000_000), features=features),
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=1_000)
    args = parser.parse_args()
    ds = make_dataset_dict(args.n_train, args.n_val, args.n_test)
    ds.push_to_hub(args.name)
