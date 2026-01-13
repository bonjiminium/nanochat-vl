"""
Common utilities for nanochat-vl.
"""

import os

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
