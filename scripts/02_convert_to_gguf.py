#!/usr/bin/env python3
"""
Step 2: Convert a HuggingFace model to GGUF format.

=== What is GGUF? ===
GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp.
It packs everything needed for inference into one file:
  - Model architecture metadata (layer count, hidden size, etc.)
  - Tokenizer vocabulary and merge rules
  - All weight tensors in a specified precision

Think of it like converting a video from .avi to .mp4 — same content, 
different container format optimized for a specific player (llama.cpp).

=== Why convert to F16 first? ===
We convert to F16 (half precision) as our baseline because:
  1. It's already 2x smaller than FP32 with virtually no quality loss
  2. Most inference is done at F16 or lower anyway
  3. It gives us a clean starting point for further quantization
  4. The quantize tool works best when starting from F16

=== What this script does ===
1. Checks if llama.cpp is cloned and built (clones/builds if needed)
2. Runs convert_hf_to_gguf.py to convert the model
3. Outputs a single .gguf file at F16 precision
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_models_dir, get_model_size, format_size
from src.quantizer import GGUFQuantizer


def convert_model(model_dir: Path, output_path: Path) -> bool:
    """
    Convert a HuggingFace model to GGUF format.
    
    Args:
        model_dir: Path to the HuggingFace model directory
        output_path: Path where the GGUF file will be saved
        
    Returns:
        True if conversion succeeded
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    
    # Initialize the quantizer (handles llama.cpp setup)
    quantizer = GGUFQuantizer()
    
    # Step 1: Ensure llama.cpp is ready
    if not quantizer.ensure_llama_cpp():
        print("\n🔧 Setting up llama.cpp (one-time setup)...")
        print("   This clones the repo and compiles the C++ tools.")
        print("   It may take a few minutes.\n")
        
        if not quantizer.clone_and_build():
            print("❌ Failed to set up llama.cpp")
            return False
    
    # Step 2: Convert to GGUF
    print(f"\n📋 Conversion Plan:")
    print(f"   Input:  {model_dir} (HuggingFace format)")
    print(f"   Output: {output_path} (GGUF F16 format)")
    print()
    
    # Educational: explain the conversion process
    print("💡 What happens during conversion:")
    print("   1. Read model config (architecture, layer sizes, etc.)")
    print("   2. Read tokenizer (vocabulary, merge rules)")  
    print("   3. Read weight tensors from .safetensors files")
    print("   4. Convert weights to F16 (half precision)")
    print("   5. Pack everything into a single .gguf file")
    print()
    
    success = quantizer.convert_hf_to_gguf(model_dir, output_path)
    
    if success:
        size_mb = get_model_size(output_path)
        print(f"\n🎉 Success! Created: {output_path}")
        print(f"   Size: {format_size(int(size_mb * 1024 * 1024))}")
        print(f"\n🎯 Next step: Quantize to different bit-widths")
        print(f"   python scripts/03_quantize.py --input {output_path}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to GGUF format (F16 baseline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_convert_to_gguf.py
  python 02_convert_to_gguf.py --model-dir models/original/SmolLM2-135M
  python 02_convert_to_gguf.py --output models/quantized/my-model-F16.gguf
        """
    )
    
    dirs = get_models_dir()
    default_model_dir = dirs["original"] / "SmolLM2-135M"
    default_output = dirs["quantized"] / "model-F16.gguf"
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(default_model_dir),
        help=f"Path to HuggingFace model directory (default: {default_model_dir})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output GGUF file path (default: {default_output})"
    )
    
    args = parser.parse_args()
    
    success = convert_model(Path(args.model_dir), Path(args.output))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
