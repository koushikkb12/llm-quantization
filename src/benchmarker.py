"""
Model Benchmarking Utilities
=============================

This module provides tools for benchmarking quantized GGUF models to measure
the real-world trade-offs of quantization:

- **Model size** — How much disk space / memory does each variant need?
- **Inference speed** — How many tokens per second can the model generate?
- **Load time** — How quickly does the model initialise?
- **Memory usage** — What is the peak RAM consumption during inference?

Why Benchmark Quantized Models?
    Quantization is fundamentally a **quality vs. efficiency trade-off**.
    Lower-bit models are smaller and faster, but may produce lower-quality
    outputs.  Benchmarking helps you find the sweet spot for your specific
    use case:

    - **Edge / mobile deployment** → prioritise size (Q4_K_M or Q2_K)
    - **Latency-sensitive APIs** → prioritise tokens/sec (Q4_K_M or Q5_K_M)
    - **Quality-critical tasks** → prioritise accuracy (Q8_0 or F16)

Dependencies:
    - ``llama-cpp-python`` — Python bindings for llama.cpp inference
    - ``tabulate`` — Pretty-printed comparison tables
    - ``matplotlib`` — Visualisation of benchmark results
"""

import json
import logging
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
# Use a non-interactive backend so we can save plots without a display.
# This is essential for headless servers and CI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llama_cpp import Llama
from tabulate import tabulate

from .utils import get_model_size

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Benchmark GGUF models to compare quantization trade-offs.

    This class loads GGUF models via ``llama-cpp-python``, runs inference with
    a given prompt, and collects performance metrics including timing, memory
    usage, and throughput.

    Attributes:
        n_ctx (int): Context window size (in tokens).  Larger values let the
            model consider more text but use more memory.
        n_threads (int): Number of CPU threads for inference.  Usually best
            set to the number of *physical* cores (not hyper-threads).
        n_gpu_layers (int): Number of model layers to offload to GPU.
            Set to -1 to offload ALL layers (recommended when GPU is available).
            Set to 0 to use CPU only.

    Example::

        benchmarker = ModelBenchmarker(n_ctx=512, n_threads=8, n_gpu_layers=-1)

        result = benchmarker.benchmark_model(
            model_path=Path("models/quantized/SmolLM2-135M-Q4_K_M.gguf"),
            prompt="The meaning of life is",
            n_tokens=100,
        )
        print(f"Speed: {result['tokens_per_sec']:.1f} tok/s")
    """

    def __init__(self, n_ctx: int = 512, n_threads: int = 8, n_gpu_layers: int = -1) -> None:
        """Initialise the benchmarker.

        Args:
            n_ctx: Context window size in tokens.  512 is a sensible default
                   for quick benchmarks; increase for longer generation tasks.
            n_threads: Number of CPU threads to use during inference.
                       Set this to your machine's physical core count for
                       optimal performance.
            n_gpu_layers: Number of layers to offload to GPU. -1 means all
                         layers (fully GPU-accelerated). 0 means CPU only.
                         With an L4 GPU (24GB VRAM), you can easily offload
                         all layers of models up to ~13B parameters.
        """
        self.n_ctx: int = n_ctx
        self.n_threads: int = n_threads
        self.n_gpu_layers: int = n_gpu_layers

        # Auto-detect GPU
        self._gpu_available = self._check_gpu()
        if self.n_gpu_layers != 0 and self._gpu_available:
            logger.info("🚀 GPU detected — offloading %s layers",
                       "ALL" if self.n_gpu_layers == -1 else str(self.n_gpu_layers))
        elif self.n_gpu_layers != 0:
            logger.info("💻 No GPU detected — running on CPU only")
            self.n_gpu_layers = 0

    @staticmethod
    def _check_gpu() -> bool:
        """Check if a CUDA GPU is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Single-model benchmark
    # ──────────────────────────────────────────────────────────────────────

    def benchmark_model(
        self,
        model_path: Path,
        prompt: str,
        n_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark a single GGUF model file.

        This method:
        1. Records the model's file size on disk.
        2. Loads the model and measures **load time**.
        3. Runs text generation and measures **tokens per second**.
        4. Records **peak memory** usage (via the ``resource`` module on Linux).
        5. Returns all metrics in a structured dictionary.

        **How tokens/sec is calculated:**
            We measure the wall-clock time for the ``create_completion`` call,
            then divide the number of requested tokens by that time.  This
            gives the *generation throughput*, which is the metric most
            relevant to end-user experience.

        **Memory measurement:**
            On Linux, ``resource.getrusage(resource.RUSAGE_SELF).ru_maxrss``
            gives the peak resident set size (RSS) in **kilobytes**.  We
            convert this to megabytes for readability.  Note that this is a
            high-water mark for the entire process, so the first model
            benchmarked in a session will be the most accurate.

        Args:
            model_path: Absolute or relative path to a ``.gguf`` model file.
            prompt: The text prompt to feed to the model.
            n_tokens: Maximum number of tokens to generate.

        Returns:
            dict: Benchmark results with the following keys:

            - ``model_name`` (str): Filename of the model.
            - ``model_size_mb`` (float): File size in megabytes.
            - ``load_time_s`` (float): Time to load the model, in seconds.
            - ``tokens_per_sec`` (float): Generation throughput.
            - ``total_time_s`` (float): Total wall-clock time (load + generate).
            - ``peak_memory_mb`` (float): Peak process memory in megabytes.
            - ``output_text`` (str): The generated text.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            Exception: Propagates any errors from llama-cpp-python.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_name = model_path.name
        model_size_mb = get_model_size(model_path)

        logger.info("📊 Benchmarking: %s (%.1f MB)", model_name, model_size_mb)

        # ── Step 1: Load the model ───────────────────────────────────────
        # Measuring load time is important because larger (higher-bit) models
        # take longer to read from disk and initialise in memory.
        load_start = time.perf_counter()

        llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,  # Offload layers to GPU for acceleration
            verbose=False,  # Suppress llama.cpp's own logging
        )

        load_time = time.perf_counter() - load_start

        # ── Step 2: Run inference ────────────────────────────────────────
        # We time only the generation step (not loading) to get a clean
        # throughput measurement.
        gen_start = time.perf_counter()

        output = llm.create_completion(
            prompt=prompt,
            max_tokens=n_tokens,
            echo=False,  # Don't include the prompt in the output
        )

        gen_time = time.perf_counter() - gen_start

        # ── Step 3: Collect metrics ──────────────────────────────────────
        # Extract the generated text from the response.
        output_text = output["choices"][0]["text"] if output["choices"] else ""

        # Calculate tokens per second.
        # The actual number of tokens generated may differ from n_tokens if
        # the model hits an EOS token early.
        actual_tokens = output["usage"]["completion_tokens"]
        tokens_per_sec = actual_tokens / gen_time if gen_time > 0 else 0.0

        # Peak memory usage (Linux-specific).
        # ru_maxrss is in kilobytes on Linux, bytes on macOS.
        peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_mb = peak_memory_kb / 1024  # Convert KB → MB

        total_time = load_time + gen_time

        result: Dict[str, Any] = {
            "model_name": model_name,
            "model_size_mb": model_size_mb,
            "load_time_s": round(load_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_time_s": round(total_time, 3),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "output_text": output_text,
        }

        logger.info(
            "   ✅ Done — %.1f tok/s | load %.2fs | %.1f MB peak RAM",
            tokens_per_sec, load_time, peak_memory_mb,
        )

        # Explicitly delete the model to free memory before loading the next.
        del llm

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Multi-model benchmark
    # ──────────────────────────────────────────────────────────────────────

    def benchmark_all(
        self,
        models_dir: Path,
        prompt: str,
        n_tokens: int = 100,
    ) -> List[Dict[str, Any]]:
        """Benchmark every GGUF model in a directory.

        This scans ``models_dir`` for all files matching ``*.gguf``, sorts them
        by size (largest first — typically the highest-precision model), and
        benchmarks each one sequentially.

        Running benchmarks sequentially (rather than in parallel) ensures that
        memory measurements aren't contaminated by concurrent model loads.

        Args:
            models_dir: Directory containing ``.gguf`` model files.
            prompt: The text prompt to use for all benchmarks.
            n_tokens: Maximum number of tokens to generate per benchmark.

        Returns:
            List[dict]: A list of result dictionaries (one per model), each
            with the same structure as :meth:`benchmark_model` returns.
        """
        # Find all GGUF files and sort by size (descending) so the largest
        # (highest-quality) model is benchmarked first.
        gguf_files = sorted(
            models_dir.glob("*.gguf"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )

        if not gguf_files:
            logger.warning("⚠️  No .gguf files found in %s", models_dir)
            return []

        logger.info(
            "🔍 Found %d GGUF model(s) in %s", len(gguf_files), models_dir
        )

        results: List[Dict[str, Any]] = []

        for i, model_path in enumerate(gguf_files, start=1):
            logger.info(
                "\n── Model %d / %d ──────────────────────────────────────",
                i, len(gguf_files),
            )
            try:
                result = self.benchmark_model(model_path, prompt, n_tokens)
                results.append(result)
            except Exception as exc:
                logger.error("❌ Failed to benchmark %s: %s", model_path.name, exc)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Output & visualisation
    # ──────────────────────────────────────────────────────────────────────

    def print_comparison(self, results: List[Dict[str, Any]]) -> None:
        """Print a formatted comparison table of benchmark results.

        Uses the ``tabulate`` library to produce a clean, aligned table that
        makes it easy to compare model variants side by side.

        The table includes:
        - Model name
        - File size (MB)
        - Load time (seconds)
        - Inference speed (tokens/sec)
        - Total time (seconds)
        - Peak memory (MB)

        Args:
            results: List of benchmark result dicts from :meth:`benchmark_model`
                     or :meth:`benchmark_all`.
        """
        if not results:
            print("No benchmark results to display.")
            return

        # Prepare table rows — exclude the output_text field for readability.
        headers = [
            "Model",
            "Size (MB)",
            "Load (s)",
            "Tok/s",
            "Total (s)",
            "Peak RAM (MB)",
        ]

        rows = [
            [
                r["model_name"],
                f"{r['model_size_mb']:.1f}",
                f"{r['load_time_s']:.3f}",
                f"{r['tokens_per_sec']:.2f}",
                f"{r['total_time_s']:.3f}",
                f"{r['peak_memory_mb']:.1f}",
            ]
            for r in results
        ]

        print("\n" + "=" * 80)
        print("  📊  QUANTIZATION BENCHMARK RESULTS")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
        print("=" * 80 + "\n")

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save benchmark results to a JSON file.

        The JSON file contains the full results array including generated text,
        making it suitable for later analysis or comparison across runs.

        Args:
            results: List of benchmark result dicts.
            output_path: Path for the output JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("💾 Results saved to %s", output_path)

    def plot_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Create visualisation charts from benchmark results.

        Generates two publication-quality bar charts:

        1. **Model Size Comparison** — shows how file size decreases with
           more aggressive quantization.  This directly correlates with
           memory usage and download/deployment cost.

        2. **Inference Speed Comparison** — shows tokens/sec for each
           variant.  Smaller models often run *faster* because less data
           needs to move through the memory hierarchy (cache lines, RAM
           bandwidth, etc.).

        Both charts are saved as PNG files in ``output_dir``.

        Args:
            results: List of benchmark result dicts.
            output_dir: Directory to save the chart images into.
        """
        if not results:
            logger.warning("No results to plot.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data for plotting.
        model_names = [r["model_name"].replace(".gguf", "") for r in results]
        sizes = [r["model_size_mb"] for r in results]
        speeds = [r["tokens_per_sec"] for r in results]

        # Use a pleasant colour palette.
        # Colors progress from cool (blue/green) for large models to warm
        # (orange/red) for small models, visually reinforcing the
        # size-vs-compression trade-off.
        colors = plt.cm.viridis([i / max(len(results) - 1, 1) for i in range(len(results))])

        # ── Chart 1: Model Size ──────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        bars1 = ax1.bar(model_names, sizes, color=colors, edgecolor="white", linewidth=0.5)

        ax1.set_title("Model Size Comparison", fontsize=16, fontweight="bold", pad=15)
        ax1.set_xlabel("Model Variant", fontsize=12)
        ax1.set_ylabel("Size (MB)", fontsize=12)
        ax1.tick_params(axis="x", rotation=45, labelsize=10)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on top of each bar.
        for bar, size in zip(bars1, sizes):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sizes) * 0.01,
                f"{size:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        fig1.tight_layout()
        size_chart_path = output_dir / "model_size_comparison.png"
        fig1.savefig(size_chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)

        logger.info("📈 Size chart saved to %s", size_chart_path)

        # ── Chart 2: Inference Speed ─────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        bars2 = ax2.bar(model_names, speeds, color=colors, edgecolor="white", linewidth=0.5)

        ax2.set_title("Inference Speed Comparison", fontsize=16, fontweight="bold", pad=15)
        ax2.set_xlabel("Model Variant", fontsize=12)
        ax2.set_ylabel("Tokens per Second", fontsize=12)
        ax2.tick_params(axis="x", rotation=45, labelsize=10)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on top of each bar.
        for bar, speed in zip(bars2, speeds):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speeds) * 0.01,
                f"{speed:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        fig2.tight_layout()
        speed_chart_path = output_dir / "inference_speed_comparison.png"
        fig2.savefig(speed_chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        logger.info("📈 Speed chart saved to %s", speed_chart_path)
