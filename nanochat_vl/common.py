"""
Common utilities for nanochat-vl.
"""

import os, requests

def download_file(url, path):
    if os.path.exists(path): return path
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    print(f"Downloaded to {path}")
    return path

def get_base_dir():
    """Get the base directory for all nanochat-vl artifacts."""
    if os.environ.get("NANOCHAT_VL_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_VL_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat-vl")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir
