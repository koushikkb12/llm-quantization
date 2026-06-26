"""
Shared utilities for the LLM Quantization Lab.

This module provides common constants, paths, and helper functions
used throughout the project.
"""

from pathlib import Path
from collections import OrderedDict


# ============================================================================
# Constants
# ============================================================================

# Default model to download — SmolLM2-135M is tiny (~270MB in FP32) and
# perfect for learning quantization without needing massive storage/compute.
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

# Quantization types available in llama.cpp, ordered from highest to lowest
# precision. Each type represents a different tradeoff between model size
# and output quality.
#
# How quantization types are named:
#   - The number (2, 4, 5, 8) = bits per weight
#   - 'K' = k-quant method (uses different precision for different layers —
#           attention layers get higher precision, feed-forward gets lower)
#   - 'M' = medium preset (balances speed and quality)
#   - 'S' = small (more aggressive compression)
#   - 'L' = large (less compression, higher quality)
#   - '0' = basic/legacy method (uniform quantization across all layers)

QUANT_TYPES = OrderedDict([
    ("F16", "Half precision (16-bit float) — baseline, no quantization"),
    ("Q8_0", "8-bit quantization — minimal quality loss, ~2x compression"),
    ("Q5_K_M", "5-bit k-quant (medium) — good balance of size and quality"),
    ("Q4_K_M", "4-bit k-quant (medium) — most popular for deployment, ~4x compression"),
    ("Q4_0", "4-bit basic — faster quantization, slightly lower quality than Q4_K_M"),
    ("Q2_K", "2-bit k-quant — extreme compression (~8x), noticeable quality loss"),
])

# Quantization types to actually quantize (everything except F16, which is the baseline)
QUANT_TYPES_TO_RUN = [k for k in QUANT_TYPES if k != "F16"]


# ============================================================================
# Path helpers
# ============================================================================

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    Works by navigating up from this file's location (src/utils.py → project root).
    """
    return Path(__file__).parent.parent.resolve()


def get_models_dir() -> dict:
    """
    Returns a dict with paths to model directories.
    
    Returns:
        dict with keys:
            - 'original': Path to directory for downloaded HuggingFace models
            - 'quantized': Path to directory for GGUF quantized models
    """
    root = get_project_root()
    return {
        "original": root / "models" / "original",
        "quantized": root / "models" / "quantized",
    }


def get_llama_cpp_dir() -> Path:
    """Returns the path to the llama.cpp directory."""
    return get_project_root() / "llama.cpp"


# ============================================================================
# File size helpers
# ============================================================================

def get_model_size(path: Path) -> float:
    """
    Returns the file size of a model in megabytes (MB).
    
    Args:
        path: Path to the model file
        
    Returns:
        Size in MB as a float
        
    Why this matters for quantization:
        The whole point of quantization is to reduce model size.
        A model in FP32 uses 4 bytes per parameter.
        A model in FP16 uses 2 bytes per parameter (2x smaller).
        A model in Q4 uses ~0.5 bytes per parameter (8x smaller!).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path.stat().st_size / (1024 * 1024)


def format_size(size_bytes: int) -> str:
    """
    Converts bytes to a human-readable string (e.g., '1.23 GB').
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_model_sizes(num_params: int) -> dict:
    """
    Estimates model size at different precisions given parameter count.
    
    This is a key insight for understanding quantization:
    Each parameter is stored as a number, and the 'precision' determines
    how many bits (and thus bytes) are used for each parameter.
    
    Args:
        num_params: Number of model parameters (e.g., 135_000_000 for 135M)
        
    Returns:
        Dict mapping precision name to estimated size in MB
    """
    # Bytes per parameter at each precision level
    bytes_per_param = {
        "FP32 (32-bit)": 4.0,
        "FP16 (16-bit)": 2.0,
        "INT8 (8-bit)": 1.0,
        "INT4 (4-bit)": 0.5,
        "INT2 (2-bit)": 0.25,
    }
    
    return {
        name: (num_params * bpp) / (1024 * 1024)
        for name, bpp in bytes_per_param.items()
    }
