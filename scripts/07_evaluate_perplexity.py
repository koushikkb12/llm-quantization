#!/usr/bin/env python3
"""
Step 7: Evaluate model perplexity to measure quantization quality.

=== What is Perplexity? ===
Perplexity measures how "surprised" a model is by text it hasn't seen.
Lower perplexity = better quality. Think of it as:
  - Perplexity 5: The model predicts the next word from ~5 equally likely options
  - Perplexity 10: The model predicts from ~10 options (less certain)
  - Perplexity 100: The model is very uncertain (poor quality)

=== Why measure perplexity? ===
Quantization reduces model size but may degrade quality. Perplexity gives
us an objective number to compare:
  - F16 baseline -> perplexity X
  - Q4_K_M -> perplexity X+0.1 (negligible degradation)
  - Q2_K -> perplexity X+2.0 (significant degradation)

This helps users choose the right quantization level for their needs.

=== How we measure it ===
We use llama-perplexity from llama.cpp, which:
  1. Loads the GGUF model
  2. Feeds WikiText-2 test data through the model
  3. Calculates cross-entropy loss on predicting each token
  4. Reports the final perplexity score
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantizer import GGUFQuantizer
from src.utils import get_model_size, get_project_root

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model perplexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--model", type=str, help="Path to single GGUF file")
    parser.add_argument("--models-dir", type=str, help="Directory with multiple GGUF files to evaluate all")
    parser.add_argument("--ctx-size", type=int, default=512, help="Context size (default: 512)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (default: -1)")
    
    default_out = get_project_root() / "results" / "perplexity_results.json"
    parser.add_argument("--output", type=str, default=str(default_out), help=f"Output JSON file (default: {default_out})")
    
    args = parser.parse_args()
    
    if not args.model and not args.models_dir:
        print("❌ Error: Must specify either --model or --models-dir")
        sys.exit(1)
        
    quantizer = GGUFQuantizer()
    models_to_test = []
    
    if args.model:
        models_to_test.append(Path(args.model))
    
    if args.models_dir:
        d = Path(args.models_dir)
        if d.exists():
            models_to_test.extend([f for f in d.glob("*.gguf") if f not in models_to_test])
            
    # Sort by size descending
    models_to_test.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    print("\n" + "="*80)
    print(f"  📏 Evaluating Perplexity on {len(models_to_test)} model(s)")
    print("="*80 + "\n")
    
    results = []
    for m in models_to_test:
        size_mb = get_model_size(m)
        print(f"\nProcessing {m.name} ({size_mb:.1f} MB)...")
        try:
            res = quantizer.evaluate_perplexity(
                m,
                n_ctx=args.ctx_size,
                n_batch=args.batch_size,
                n_gpu_layers=args.gpu_layers
            )
            res["size_mb"] = round(size_mb, 1)
            results.append(res)
        except Exception as e:
            print(f"❌ Error evaluating {m.name}: {e}")
            
    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    # Print table
    try:
        from tabulate import tabulate
        rows = [[r["model_name"], f"{r['size_mb']} MB", r["perplexity"]] for r in results]
        print("\n" + tabulate(rows, headers=["Model", "Size", "Perplexity"], tablefmt="rounded_grid"))
    except ImportError:
        print("\nModel | Size | Perplexity")
        print("-" * 40)
        for r in results:
            print(f"{r['model_name']} | {r['size_mb']} MB | {r['perplexity']}")
            
    print(f"\n✅ Results saved to {out_path}")

if __name__ == "__main__":
    main()
