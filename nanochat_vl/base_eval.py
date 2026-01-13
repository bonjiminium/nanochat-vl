"Evaluate CORE metric for a given model."
import os, csv, time, json, yaml, random, zipfile, tempfile, shutil
from nanochat_vl.common import get_base_dir, download_file
from nanochat_vl.core_eval import evaluate_task

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref: zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)
    print(f"Placed eval_bundle at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    "Evaluate a base model on the CORE benchmark."
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        zip_path = os.path.join(base_dir, "eval_bundle.zip")
        download_file(EVAL_BUNDLE_URL, zip_path)
        place_eval_bundle(zip_path)
    with open(os.path.join(eval_bundle_dir, "core.yaml"), 'r') as f: config = yaml.safe_load(f)
    tasks = config['icl_tasks']
    random_baselines = {}
    with open(os.path.join(eval_bundle_dir, "eval_meta_data.csv"), 'r') as f:
        for row in csv.DictReader(f): random_baselines[row['Eval Task']] = float(row['Random baseline'])
    results, centered_results = {}, {}
    for task in tasks:
        t0 = time.time()
        label = task['label']
        task_meta = dict(task_type=task['icl_task_type'], dataset_uri=task['dataset_uri'], num_fewshot=task['num_fewshot'][0], continuation_delimiter=task.get('continuation_delimiter', ' '))
        print(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, {task_meta['task_type']})... ", end='', flush=True)
        with open(os.path.join(eval_bundle_dir, "eval_data", task_meta['dataset_uri']), 'r') as f: data = [json.loads(line) for line in f]
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0: data = data[:max_per_task]
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        baseline = random_baselines[label]
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        centered_results[label] = centered
        print(f"acc: {accuracy:.4f} | centered: {centered:.4f} | {time.time() - t0:.1f}s")
    core_metric = sum(centered_results.values()) / len(centered_results)
    print(f"CORE metric: {core_metric:.4f}")
    from nanochat_vl.report import get_report
    get_report().log(section="Base model evaluation", data=[dict(core_metric=core_metric), centered_results])
    return dict(results=results, centered_results=centered_results, core_metric=core_metric)
