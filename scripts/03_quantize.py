#!/usr/bin/env python3
"""
Step 3: Quantize the GGUF model to different bit-widths.

This is where the magic happens! We take our F16 baseline GGUF and create
quantized versions at different precision levels.

=== How llama.cpp quantization works ===

For basic types (Q4_0, Q8_0):
  - Weights are divided into blocks (typically 32 values)
  - For each block, compute min/max values
  - Calculate scale = (max - min) / (2^bits - 1)
  - Map each float to the nearest integer: q = round((w - min) / scale)
  - Store: [scale, min/zero_point, quantized_values...]

For K-quant types (Q4_K_M, Q5_K_M, Q2_K):
  - Uses a more sophisticated "super-block" structure
  - Different layers get different precision:
    * Attention layers → higher precision (more sensitive to errors)
    * Feed-forward layers → lower precision (more tolerant)
  - The 'M' suffix means "medium" — a balanced preset
  - This is why Q4_K_M generally outperforms Q4_0 at the same size

=== Expected compression ratios (vs F16) ===
  Q8_0:   ~2x compression
  Q5_K_M: ~3x compression
  Q4_K_M: ~4x compression
  Q4_0:   ~4x compression
  Q2_K:   ~6-8x compression

=== Importance Matrix (imatrix) Calibration ===
For MoE and large models, pass --imatrix to use a pre-computed importance
matrix. This tells the quantizer which weights matter most, resulting in
significantly better quality — especially at lower bit-widths (Q2, Q3, IQ).

Generate an imatrix with: python scripts/06_generate_imatrix.py
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_models_dir, get_model_size, format_size, QUANT_TYPES, QUANT_TYPES_TO_RUN, PRODUCTION_QUANT_LADDER
from src.quantizer import GGUFQuantizer


def quantize_all(input_gguf: Path, output_dir: Path, quant_types: list, imatrix_path=None) -> dict:
    """
    Quantize a GGUF model to multiple bit-widths.
    
    Args:
        input_gguf: Path to the F16 GGUF file
        output_dir: Directory to save quantized models
        quant_types: List of quantization types to create
        
    Returns:
        Dict mapping quant_type -> output_path for successful quantizations
    """
    quantizer = GGUFQuantizer()
    
    # Verify llama.cpp is ready
    if not quantizer.ensure_llama_cpp():
        print("❌ llama.cpp not ready. Run 02_convert_to_gguf.py first.")
        return {}
    
    # Print educational info about each quant type
    print(f"\n💡 Quantization types we'll create:\n")
    for qt in quant_types:
        desc = QUANT_TYPES.get(qt, "Unknown")
        print(f"   {qt:>8s}: {desc}")
        
    if imatrix_path:
        print(f"\n   🔬 Using importance matrix: {imatrix_path}")
        print(f"   This will produce higher-quality quantizations,")
        print(f"   especially for MoE models and low-bit (Q2/Q3/IQ) types.")
    print()
    
    # Run quantization for each type
    results = quantizer.quantize_all(input_gguf, output_dir, quant_types, imatrix_path=imatrix_path)
    
    # Print final comparison table
    if results:
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        
        input_size = get_model_size(input_gguf)
        
        rows = []
        for qt, path in results.items():
            size = get_model_size(path)
            compression = input_size / size if size > 0 else 0
            saving = ((input_size - size) / input_size) * 100
            rows.append([
                qt,
                QUANT_TYPES.get(qt, ""),
                f"{size:.1f} MB",
                f"{compression:.1f}x",
                f"{saving:.0f}%",
            ])
        
        print(f"\n{'='*80}")
        print(f"  📊 Quantization Results Summary")
        print(f"  Baseline: F16 = {input_size:.1f} MB")
        print(f"{'='*80}\n")
        
        if tabulate:
            headers = ["Type", "Description", "Size", "Compression", "Size Saved"]
            print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))
        else:
            for row in rows:
                print(f"  {row[0]:>8s} | {row[2]:>10s} | {row[3]:>6s} | {row[4]:>5s}")
        
        print(f"\n🎯 Next step: Benchmark all models")
        print(f"   python scripts/04_benchmark.py")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a GGUF model to various bit-widths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available quantization types:
  Q8_0   — 8-bit, minimal quality loss
  Q5_K_M — 5-bit k-quant, good balance
  Q4_K_M — 4-bit k-quant, most popular
  Q4_0   — 4-bit basic, fast
  Q2_K   — 2-bit, extreme compression

Examples:
  python 03_quantize.py                                    # Quantize to default types
  python 03_quantize.py --types Q4_K_M Q8_0                # Only specific types
  python 03_quantize.py --production                        # Full production ladder (12 types)
  python 03_quantize.py --imatrix path/to/imatrix.dat       # Use importance matrix
  python 03_quantize.py --production --imatrix imatrix.dat  # Production + imatrix (best quality)
        """
    )
    
    dirs = get_models_dir()
    default_input = dirs["quantized"] / "model-F16.gguf"
    
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help=f"Input F16 GGUF file (default: {default_input})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(dirs["quantized"]),
        help=f"Output directory (default: {dirs['quantized']})"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(QUANT_TYPES.keys()),
        default=QUANT_TYPES_TO_RUN,
        help="Quantization types to create (default: all)"
    )
    parser.add_argument(
        '--imatrix',
        type=str,
        default=None,
        help='Path to importance matrix file (.dat) for calibrated quantization. '
             'Strongly recommended for MoE models. Generate with 06_generate_imatrix.py'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Use the full production quantization ladder (12 types) for HuggingFace publishing'
    )
    
    args = parser.parse_args()
    
    # Determine which quant types to use
    if args.production:
        selected_types = PRODUCTION_QUANT_LADDER
    else:
        selected_types = args.types

    imatrix_path = Path(args.imatrix) if args.imatrix else None

    results = quantize_all(
        Path(args.input),
        Path(args.output_dir),
        selected_types,
        imatrix_path=imatrix_path,
    )
    
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
