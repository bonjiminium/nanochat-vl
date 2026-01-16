"Utilities for generating training report cards."

import os, datetime, subprocess, socket, platform, re, shutil
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")

def now_str(): return datetime.datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout.strip(): return result.stdout.strip()
        if result.returncode == 0: return ""
        return None
    except: return None

def get_git_info():
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False
    info['message'] = (run_command("git log -1 --pretty=%B") or "").split('\n')[0][:80]
    return info

def get_gpu_info():
    import torch
    if not torch.cuda.is_available(): return {"available": False}
    num_devices = torch.cuda.device_count()
    info = dict(available=True, count=num_devices, names=[], memory_gb=[])
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))
    info["cuda_version"] = torch.version.cuda or "unknown"
    return info

def get_system_info():
    import torch, psutil
    info = {}
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    info['user'] = os.environ.get('USER', 'unknown')
    info['working_dir'] = os.getcwd()
    return info

def estimate_cost(gpu_info, runtime_hours=None):
    gpu_hourly_rates = {'B200': 6.25, 'H200': 4.54, 'H100': 3.95, 'L40S': 1.95, 'A10': 1.10, 'L4': 0.80, 'T4': 0.59}
    default_rate = 3.0
    if not gpu_info.get("available"): return None
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    if 'A100' in gpu_name:
        hourly_rate = (2.50 if '80GB' in gpu_name else 2.10) * gpu_info["count"]
    else:
        for gpu_type, rate in gpu_hourly_rates.items():
            if gpu_type in gpu_name:
                hourly_rate = rate * gpu_info["count"]
                break
    if hourly_rate is None: hourly_rate = default_rate * gpu_info["count"]
    return dict(hourly_rate=hourly_rate, gpu_type=gpu_name, estimated_total=hourly_rate * runtime_hours if runtime_hours else None)

def get_dep_count(): return int(run_command("pip list --format=freeze | wc -l") or 0)

def get_bloat_info():
    extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
    git_patterns = ' '.join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or '').split('\n') if f]
    num_files = len(file_list)
    num_lines, num_chars = 0, 0
    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            parts = wc_output.strip().split('\n')[-1].split()
            if len(parts) >= 2: num_lines, num_chars = int(parts[0]), int(parts[1])
    num_tokens = num_chars // 4
    return dict(num_files=num_files, num_lines=num_lines, num_chars=num_chars, num_tokens=num_tokens)

def get_dep_count(): return int(run_command("pip list --format=freeze | wc -l") or 0)

def extract(section, keys):
    if not isinstance(keys, list): keys = [keys]
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line: out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line[len(prefix):].strip()
            try: return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except: pass
    return None

def generate_header(git_info, bloat_info, gpu_info, sys_info, cost_info, dep_count):
    header = f"""# nanochat-vl training report

Generated: {now_str()}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""
    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"
    if cost_info and cost_info.get("hourly_rate", 0) > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""
    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

### Bloat
- Characters: {bloat_info['num_chars']:,}
- Lines: {bloat_info['num_lines']:,}
- Files: {bloat_info['num_files']:,}
- Tokens (approx): {bloat_info['num_tokens']:,}
- Dependencies: {dep_count}

"""
    return header

def slugify(text): return text.lower().replace(" ", "-")

EXPECTED_FILES = [
    "tokenizer-training.md", "tokenizer-evaluation.md",
    "base-model-training.md", "base-model-loss.md", "base-model-evaluation.md",
    "midtraining.md", "chat-evaluation-mid.md", "chat-sft.md",
    "chat-evaluation-sft.md", "chat-rl.md", "chat-evaluation-rl.md",
    "vl-training.md", "vl-evaluation.md",
]

class Report:
    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def generate(self):
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {}
        start_time, end_time = None, None
        chat_metrics = ["MMLU", "ARC", "GSM8K", "HumanEval", "ChatCORE"]
        with open(report_file, "w", encoding="utf-8") as out_file:
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started: ")
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run reset?")
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file: section = in_file.read()
                if "rl" not in file_name: end_time = extract_timestamp(section, "timestamp: ")
                if file_name == "base-model-evaluation.md": final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md": final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md": final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md": final_metrics["rl"] = extract(section, "GSM8K")
                out_file.write(section)
                out_file.write("\n")
            out_file.write("## Summary\n\n")
            out_file.write(bloat_data)
            out_file.write("\n\n")
            all_metrics = set()
            for stage_metrics in final_metrics.values(): all_metrics.update(stage_metrics.keys())
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            stages = ["base", "mid", "sft", "rl"]
            metric_width, value_width = 15, 8
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages: header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages: separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages: row += f" {str(final_metrics.get(stage, {}).get(metric, '-')).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            print(f"DEBUG: start_time={start_time}, end_time={end_time}")
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours, minutes = total_seconds // 3600, (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self, git_info, bloat_info, gpu_info, sys_info, cost_info, dep_count):
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path): os.remove(file_path)
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file): os.remove(report_file)
        header = generate_header(git_info, bloat_info, gpu_info, sys_info, cost_info, dep_count)
        header_file = os.path.join(self.report_dir, "header.md")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {now_str()}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

    def log(self, section, data):
        slug = slugify(section)
        path = os.path.join(self.report_dir, f"{slug}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {now_str()}\n\n")
            for item in data:
                if not item: continue
                if isinstance(item, str): f.write(item + "\n")
                else:
                    for k, v in item.items():
                        vstr = f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) and v >= 10000 else str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return path

def get_report():
    from nanochat_vl.common import get_base_dir
    return Report(os.path.join(get_base_dir(), "report"))
