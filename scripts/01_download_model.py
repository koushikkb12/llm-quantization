#!/usr/bin/env python3
"""
Step 1: Download a small LLM from HuggingFace.

This script downloads SmolLM2-135M — a tiny but functional language model
that's perfect for learning quantization. At only ~135M parameters, it:
  - Downloads quickly (~270MB in FP32, ~270MB in safetensors)
  - Quantizes fast (minutes, not hours)
  - Still produces coherent text output
  - Lets you clearly see quality differences between quantization levels

=== Understanding Model Size ===
A model's size depends on two things:
  1. Number of parameters (135M in this case)
  2. Precision of each parameter:
     - FP32: 4 bytes/param → 135M × 4 = ~540 MB
     - FP16: 2 bytes/param → 135M × 2 = ~270 MB
     - INT8: 1 byte/param  → 135M × 1 = ~135 MB
     - INT4: 0.5 bytes/param → 135M × 0.5 = ~68 MB
"""

import sys
import argparse
from pathlib import Path

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import DEFAULT_MODEL_ID, get_models_dir, format_size, estimate_model_sizes


def download_model(model_id: str, output_dir: Path) -> Path:
    """
    Download a model from HuggingFace Hub.
    
    Uses snapshot_download which downloads all files (model weights,
    config, tokenizer) into a local directory.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'HuggingFaceTB/SmolLM2-135M')
        output_dir: Directory to save the downloaded model
        
    Returns:
        Path to the downloaded model directory
    """
    from huggingface_hub import snapshot_download
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # The model name is the last part of the ID
    model_name = model_id.split("/")[-1]
    local_dir = output_dir / model_name
    
    print(f"\n{'='*60}")
    print(f"  📥 Downloading: {model_id}")
    print(f"  📂 Destination: {local_dir}")
    print(f"{'='*60}\n")
    
    # Educational info about model sizes
    print("💡 Understanding model sizes:")
    print("   Each parameter is stored as a number.")
    print("   The 'precision' determines how many bytes each number uses:\n")
    
    # Estimate sizes for this model (135M params)
    num_params = 135_000_000  # SmolLM2-135M
    sizes = estimate_model_sizes(num_params)
    for precision, size_mb in sizes.items():
        print(f"   {precision:>20s}: {size_mb:>8.1f} MB")
    
    print(f"\n   Our goal: go from FP16 ({sizes['FP16 (16-bit)']:.0f} MB) to ")
    print(f"   INT4 ({sizes['INT4 (4-bit)']:.0f} MB) — a {sizes['FP16 (16-bit)']/sizes['INT4 (4-bit)']:.0f}x reduction!\n")
    
    # Download the model
    print("⏳ Downloading model files...\n")
    downloaded_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    
    # Show what was downloaded
    print(f"\n✅ Download complete!")
    print(f"   Location: {downloaded_path}\n")
    
    total_size = 0
    print("   Files downloaded:")
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"   {format_size(size):>12s}  {f.name}")
    
    print(f"\n   Total size: {format_size(total_size)}")
    
    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download a small LLM from HuggingFace for quantization experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_download_model.py
  python 01_download_model.py --model-id HuggingFaceTB/SmolLM2-360M
  python 01_download_model.py --output-dir /path/to/models
        """
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the model (default: models/original/)"
    )
    
    args = parser.parse_args()
    
    # Use default models directory if not specified
    output_dir = Path(args.output_dir) if args.output_dir else get_models_dir()["original"]
    
    model_path = download_model(args.model_id, output_dir)
    
    print(f"\n🎯 Next step: Convert to GGUF format")
    print(f"   python scripts/02_convert_to_gguf.py --model-dir {model_path}")


if __name__ == "__main__":
    main()
