#!/usr/bin/env python3
"""
Step 6: Generate an Importance Matrix (imatrix) for high-quality quantization.

=== What is an Importance Matrix? ===
An importance matrix (imatrix) measures how important each weight is to the
model's output quality. It's created by running calibration data through the
model and tracking which weights have the most impact.

=== Why imatrix matters ===
For standard dense models, k-quant heuristics (Q4_K_M, etc.) do a good job
of deciding which layers get higher/lower precision. But for:
  - MoE (Mixture of Experts) models — experts are activated sparsely,
    so heuristics fail to capture which expert weights matter most
  - Very large models (70B+) — more layers means more room for error
  - Low-bit quantization (Q2, Q3, IQ) — every bit matters more

imatrix-calibrated quantization consistently produces better results.

=== How it works ===
1. Feed calibration text through the model
2. For each weight, measure its "importance" (gradient magnitude)
3. Save this importance data to a .dat file
4. Pass the .dat file to llama-quantize, which uses it to allocate
   bits more intelligently — important weights get more precision
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantizer import GGUFQuantizer

def main():
    parser = argparse.ArgumentParser(
        description="Generate an importance matrix for calibrated quantization.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the F16 GGUF file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .dat file (default: same dir as model, model-imatrix.dat)"
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Optional path to calibration text file (default: downloads wikitext)"
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=512,
        help="Context size (default: 512)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (default: -1, all)"
    )
    parser.add_argument(
        "--fit",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Whether to adjust unset arguments to fit in device memory (default: on)"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=-1,
        help="Max number of chunks to process (default: -1, all)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        sys.exit(1)
        
    output_path = Path(args.output) if args.output else model_path.with_name(f"{model_path.stem}-imatrix.dat")
    calib_file = Path(args.calibration_file) if args.calibration_file else None
    
    quantizer = GGUFQuantizer()
    
    print("\n" + "="*80)
    print("  🔬 Genering Importance Matrix (imatrix)")
    print("  This process calibrates the quantization for better quality.")
    print("="*80 + "\n")
    
    try:
        quantizer.generate_imatrix(
            model_path,
            output_path,
            calib_file,
            args.ctx_size,
            args.batch_size,
            args.gpu_layers,
            args.fit,
            args.chunks
        )
        
        print(f"\n🎯 Next step: Use this imatrix for quantization")
        print(f"   python scripts/03_quantize.py --input {model_path} --imatrix {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating imatrix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
