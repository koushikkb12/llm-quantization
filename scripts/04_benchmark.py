#!/usr/bin/env python3
"""
Step 4: Benchmark all quantized models.

This script loads each GGUF model and measures:
  - Load time: How long to load the model into memory
  - Tokens/sec: How fast the model generates text
  - Memory usage: Peak RAM consumption
  - Model size: File size on disk

=== Why benchmarking matters ===
Quantization is all about tradeoffs. By benchmarking, we can see:
  - How much faster is Q4_K_M vs Q8_0?
  - How much memory do we save with Q2_K?
  - Is the quality loss from Q2_K worth the speed gain?

These real-world numbers help you choose the right quantization
level for your specific use case.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_models_dir, get_project_root
from src.benchmarker import ModelBenchmarker


DEFAULT_PROMPT = "Explain what artificial intelligence is in simple terms."


def run_benchmarks(
    models_dir: Path,
    prompt: str,
    n_tokens: int,
    do_plot: bool
) -> list:
    """
    Benchmark all GGUF models in a directory.
    
    Args:
        models_dir: Directory containing .gguf files
        prompt: Text prompt to use for generation
        n_tokens: Number of tokens to generate per model
        do_plot: Whether to create matplotlib charts
        
    Returns:
        List of benchmark result dicts
    """
    models_dir = Path(models_dir)
    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GGUF models
    gguf_files = sorted(models_dir.glob("*.gguf"))
    
    if not gguf_files:
        print(f"❌ No .gguf files found in {models_dir}")
        print("   Run scripts 01-03 first to download and quantize a model.")
        return []
    
    print(f"\n{'='*60}")
    print(f"  📊 LLM Quantization Benchmark")
    print(f"{'='*60}")
    print(f"  Models directory: {models_dir}")
    print(f"  Models found:     {len(gguf_files)}")
    print(f"  Prompt:           \"{prompt[:50]}...\"")
    print(f"  Tokens/model:     {n_tokens}")
    print(f"{'='*60}\n")
    
    # Print what we're measuring
    print("💡 What we measure for each model:")
    print("   • Load time   — How long to initialize the model")
    print("   • Tokens/sec  — Generation speed (higher is better)")
    print("   • Peak memory — Maximum RAM used during inference")
    print("   • Model size  — File size on disk")
    print()
    
    # Run benchmarks
    benchmarker = ModelBenchmarker(n_ctx=512, n_threads=8, n_gpu_layers=-1)
    results = benchmarker.benchmark_all(models_dir, prompt, n_tokens)
    
    if not results:
        print("❌ No successful benchmarks")
        return []
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  📋 Results")
    print(f"{'='*60}\n")
    benchmarker.print_comparison(results)
    
    # Save results
    results_file = results_dir / "benchmark_results.json"
    benchmarker.save_results(results, results_file)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate plots
    if do_plot:
        print("\n📈 Generating charts...")
        benchmarker.plot_results(results, results_dir)
        print(f"   Charts saved to: {results_dir}")
    
    # Print key insights
    print(f"\n{'='*60}")
    print(f"  💡 Key Insights")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        # Find fastest and smallest
        fastest = max(results, key=lambda r: r.get("tokens_per_sec", 0))
        smallest = min(results, key=lambda r: r.get("model_size_mb", float("inf")))
        largest = max(results, key=lambda r: r.get("model_size_mb", 0))
        
        if largest["model_size_mb"] > 0:
            compression = largest["model_size_mb"] / smallest["model_size_mb"]
            print(f"  🏆 Fastest:    {fastest['model_name']} ({fastest.get('tokens_per_sec', 0):.1f} tok/s)")
            print(f"  📦 Smallest:   {smallest['model_name']} ({smallest['model_size_mb']:.1f} MB)")
            print(f"  📉 Max compression: {compression:.1f}x ({largest['model_name']} → {smallest['model_name']})")
    
    print(f"\n🎯 Next step: Chat with your favorite model!")
    print(f"   python scripts/05_chat.py --model models/quantized/model-Q4_K_M.gguf")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all quantized GGUF models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_benchmark.py                           # Benchmark all models
  python 04_benchmark.py --n-tokens 50             # Fewer tokens (faster)
  python 04_benchmark.py --plot                     # Also generate charts
  python 04_benchmark.py --prompt "Write a poem"    # Custom prompt
        """
    )
    
    dirs = get_models_dir()
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(dirs["quantized"]),
        help=f"Directory with .gguf files (default: {dirs['quantized']})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt for generation (default: '{DEFAULT_PROMPT}')"
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate per model (default: 100)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib charts"
    )
    
    args = parser.parse_args()
    
    results = run_benchmarks(
        Path(args.models_dir),
        args.prompt,
        args.n_tokens,
        args.plot
    )
    
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
