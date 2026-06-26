"""
GGUF Quantization Wrapper
=========================

This module provides a high-level Python interface around the **llama.cpp**
quantization toolchain.  It handles:

1. **Cloning & building** llama.cpp from source.
2. **Converting** Hugging Face models to the GGUF format (F16 baseline).
3. **Quantizing** GGUF models to various lower-bit representations.

How Quantization Works (simplified):
    Neural-network weights are stored as floating-point numbers (typically
    FP32 or FP16).  Quantization maps these continuous values into a smaller
    set of discrete levels that can be represented with fewer bits.

    For example, Q4_K_M uses only 4 bits per weight on average.  Since an
    FP16 weight needs 16 bits, Q4_K_M models are roughly **4× smaller** than
    their FP16 equivalents — and they load / run faster because less data has
    to move through memory.

    The *k-quant* family (Q4_K_M, Q5_K_M, …) goes further by assigning
    **different bit-widths to different layers** based on how sensitive each
    layer is to precision loss.  Attention layers typically keep higher
    precision, while feed-forward layers can tolerate more aggressive
    compression.

Dependencies:
    - ``git`` (for cloning llama.cpp)
    - ``cmake`` + C/C++ toolchain (for building llama.cpp)
    - ``python3`` (for the convert script)
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .utils import QUANT_TYPES, get_project_root

logger = logging.getLogger(__name__)


class GGUFQuantizer:
    """High-level wrapper around the llama.cpp quantization pipeline.

    This class manages the full lifecycle of GGUF quantization:

    1. **Setup** — clone and build llama.cpp if it isn't already present.
    2. **Convert** — transform a Hugging Face model directory into an F16 GGUF
       file using ``convert_hf_to_gguf.py``.
    3. **Quantize** — compress the F16 GGUF into one or more lower-bit GGUF
       files using ``llama-quantize``.

    Attributes:
        llama_cpp_dir (Path): Root directory of the llama.cpp checkout.
        convert_script (Path): Path to ``convert_hf_to_gguf.py``.
        quantize_binary (Path): Path to the compiled ``llama-quantize`` binary.

    Example::

        quantizer = GGUFQuantizer()

        # First-time setup (clones + builds llama.cpp)
        if not quantizer.ensure_llama_cpp():
            quantizer.clone_and_build()

        # Convert HF model → F16 GGUF
        quantizer.convert_hf_to_gguf(
            model_dir=Path("models/original/SmolLM2-135M"),
            output_path=Path("models/original/SmolLM2-135M-F16.gguf"),
        )

        # Quantize F16 → multiple lower-bit variants
        quantizer.quantize_all(
            input_gguf=Path("models/original/SmolLM2-135M-F16.gguf"),
            output_dir=Path("models/quantized"),
        )
    """

    def __init__(self, llama_cpp_dir: Optional[Path] = None) -> None:
        """Initialise the quantizer.

        Args:
            llama_cpp_dir: Path to an existing (or desired) llama.cpp checkout.
                           Defaults to ``<project_root>/llama.cpp``.
        """
        if llama_cpp_dir is None:
            llama_cpp_dir = get_project_root() / "llama.cpp"

        self.llama_cpp_dir: Path = llama_cpp_dir

        # The conversion script lives in the repo root.
        self.convert_script: Path = self.llama_cpp_dir / "convert_hf_to_gguf.py"

        # After building with cmake, the quantize binary ends up here.
        self.quantize_binary: Path = self.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    # ──────────────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────────────

    def ensure_llama_cpp(self) -> bool:
        """Check whether llama.cpp is cloned **and** built.

        This performs two checks:
        1. Does the convert script exist? (proves the repo is cloned)
        2. Does the quantize binary exist? (proves it was compiled)

        Returns:
            bool: ``True`` if both the conversion script and the quantize
            binary are present and ready to use.
        """
        repo_cloned = self.convert_script.exists()
        binary_built = self.quantize_binary.exists()

        if repo_cloned and binary_built:
            logger.info("✅ llama.cpp is ready at %s", self.llama_cpp_dir)
            return True

        if not repo_cloned:
            logger.info("❌ llama.cpp repo not found at %s", self.llama_cpp_dir)
        elif not binary_built:
            logger.info(
                "⚠️  llama.cpp repo found but quantize binary is missing — "
                "rebuild may be needed"
            )

        return False

    def clone_and_build(self) -> None:
        """Clone the llama.cpp repository and build it from source.

        This method performs the following steps:

        1. **git clone** — fetches the latest llama.cpp source code from GitHub.
           We use ``--depth 1`` for a shallow clone to save bandwidth.
        2. **cmake configure** — generates the build system inside
           ``llama.cpp/build/``.
        3. **cmake build** — compiles the C/C++ code, producing the
           ``llama-quantize`` binary (among others).

        The entire process typically takes 2–5 minutes depending on your
        machine and network speed.

        Raises:
            subprocess.CalledProcessError: If any build step fails.
            RuntimeError: If the quantize binary is missing after the build.
        """
        logger.info("📦 Cloning llama.cpp …")

        # ── Step 1: Clone ────────────────────────────────────────────────
        # Shallow clone (--depth 1) downloads only the latest commit, which
        # is much faster than fetching the full history.
        if not self.llama_cpp_dir.exists():
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(self.llama_cpp_dir),
                ],
                check=True,
            )
            logger.info("✅ Repository cloned to %s", self.llama_cpp_dir)
        else:
            logger.info("📂 llama.cpp directory already exists — skipping clone")

        # ── Step 2: Configure with cmake ─────────────────────────────────
        # cmake reads CMakeLists.txt and prepares a build directory.
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🔧 Configuring build with cmake …")

        # Check if a CUDA GPU is available for GPU-accelerated builds
        cmake_args = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
        try:
            gpu_check = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if gpu_check.returncode == 0:
                cmake_args.append("-DGGML_CUDA=ON")
                logger.info("🚀 GPU detected — building with CUDA support")
            else:
                logger.info("💻 No GPU detected — building CPU-only")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("💻 No GPU detected — building CPU-only")

        subprocess.run(
            cmake_args,
            cwd=str(build_dir),
            check=True,
        )

        # ── Step 3: Compile ──────────────────────────────────────────────
        # -j$(nproc) uses all available CPU cores for a faster build.
        logger.info("🔨 Building llama.cpp (this may take a few minutes) …")
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=str(build_dir),
            check=True,
        )

        # ── Verify ───────────────────────────────────────────────────────
        if not self.quantize_binary.exists():
            raise RuntimeError(
                f"Build completed but quantize binary not found at "
                f"{self.quantize_binary}.  Check the build output for errors."
            )

        logger.info("✅ llama.cpp built successfully — quantize binary ready")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Conversion
    # ──────────────────────────────────────────────────────────────────────

    def convert_hf_to_gguf(self, model_dir: Path, output_path: Path) -> Path:
        """Convert a Hugging Face model directory to GGUF format (F16).

        **What happens under the hood:**

        The ``convert_hf_to_gguf.py`` script reads the model's safetensors /
        PyTorch weight files plus its tokenizer configuration, then writes a
        single ``.gguf`` file that llama.cpp can load directly.

        The output is an **F16 (half-precision)** GGUF file — this is essentially
        a lossless format change, not a quantization step.  Think of it as
        "re-packaging" the model into a format optimised for CPU inference.

        Args:
            model_dir: Path to the downloaded Hugging Face model directory
                       (must contain ``config.json``, weight files, etc.).
            output_path: Desired path for the output ``.gguf`` file.

        Returns:
            Path: The ``output_path`` on success.

        Raises:
            FileNotFoundError: If ``model_dir`` or the convert script is missing.
            subprocess.CalledProcessError: If the conversion script fails.
        """
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if not self.convert_script.exists():
            raise FileNotFoundError(
                f"Conversion script not found: {self.convert_script}.  "
                f"Run clone_and_build() first."
            )

        # Make sure the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "🔄 Converting HF model to GGUF …\n"
            "   Source : %s\n"
            "   Output : %s",
            model_dir, output_path,
        )

        # The convert script is a Python script that uses the current Python
        # interpreter.  We pass --outtype f16 to get a half-precision baseline.
        subprocess.run(
            [
                sys.executable,               # Use the same Python that's running us
                str(self.convert_script),
                str(model_dir),
                "--outfile", str(output_path),
                "--outtype", "f16",            # Half-precision baseline
            ],
            check=True,
        )

        logger.info("✅ GGUF conversion complete → %s", output_path)
        return output_path

    # ──────────────────────────────────────────────────────────────────────
    # Quantization
    # ──────────────────────────────────────────────────────────────────────

    def quantize(self, input_gguf: Path, output_gguf: Path, quant_type: str) -> Path:
        """Quantize a GGUF model to a specific lower-bit representation.

        **What happens under the hood:**

        The ``llama-quantize`` binary reads the input GGUF (usually F16),
        iterates through every tensor (weight matrix), and re-encodes each
        one using the chosen quantization scheme.

        For k-quant methods (Q4_K_M, Q5_K_M, etc.), the tool:
        1. Analyses each layer's importance using calibration heuristics.
        2. Assigns higher bit-widths to sensitive layers (e.g., attention).
        3. Assigns lower bit-widths to less sensitive layers (e.g., FFN).

        This importance-aware approach is why k-quants retain better quality
        than their "basic" counterparts (Q4_0, Q8_0) at the same average
        bit-width.

        Args:
            input_gguf: Path to the source GGUF file (typically F16).
            output_gguf: Desired path for the quantized output file.
            quant_type: Quantization type string — must be a key in
                        :data:`~src.utils.QUANT_TYPES` (e.g., ``'Q4_K_M'``).

        Returns:
            Path: The ``output_gguf`` path on success.

        Raises:
            FileNotFoundError: If the input file or quantize binary is missing.
            ValueError: If ``quant_type`` is not recognised.
            subprocess.CalledProcessError: If quantization fails.
        """
        # ── Validation ───────────────────────────────────────────────────
        if not input_gguf.exists():
            raise FileNotFoundError(f"Input GGUF not found: {input_gguf}")

        if not self.quantize_binary.exists():
            raise FileNotFoundError(
                f"Quantize binary not found: {self.quantize_binary}.  "
                f"Run clone_and_build() first."
            )

        if quant_type not in QUANT_TYPES:
            raise ValueError(
                f"Unknown quant type '{quant_type}'.  "
                f"Valid types: {list(QUANT_TYPES.keys())}"
            )

        # Ensure output directory exists.
        output_gguf.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "⚙️  Quantizing to %s …\n"
            "   Input  : %s\n"
            "   Output : %s\n"
            "   Info   : %s",
            quant_type, input_gguf, output_gguf, QUANT_TYPES[quant_type],
        )

        # ── Run llama-quantize ───────────────────────────────────────────
        # Usage: llama-quantize <input.gguf> <output.gguf> <type>
        subprocess.run(
            [
                str(self.quantize_binary),
                str(input_gguf),
                str(output_gguf),
                quant_type,
            ],
            check=True,
        )

        logger.info("✅ Quantization complete → %s", output_gguf)
        return output_gguf

    def quantize_all(
        self,
        input_gguf: Path,
        output_dir: Path,
        quant_types: Optional[List[str]] = None,
    ) -> List[Path]:
        """Quantize a GGUF model to multiple quantization levels.

        This is a convenience wrapper around :meth:`quantize` that iterates
        through several quantization types and produces one output file per
        type.  Output filenames are derived from the input filename with the
        quantization type appended.

        By default it processes every type in :data:`~src.utils.QUANT_TYPES`
        **except** F16 (since the input is expected to already be F16).

        Args:
            input_gguf: Path to the source F16 GGUF file.
            output_dir: Directory to write quantized files into.
            quant_types: Optional list of quantization types to produce.
                         Defaults to all types in ``QUANT_TYPES`` except F16.

        Returns:
            List[Path]: Paths to all successfully created quantized files.

        Example::

            paths = quantizer.quantize_all(
                input_gguf=Path("models/original/model-F16.gguf"),
                output_dir=Path("models/quantized"),
                quant_types=["Q4_K_M", "Q8_0"],
            )
        """
        if quant_types is None:
            # Skip F16 — the input is already F16, so "quantizing" to F16
            # would just copy the file.
            quant_types = [qt for qt in QUANT_TYPES if qt != "F16"]

        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive a base name from the input file.
        # e.g. "SmolLM2-135M-F16.gguf" → "SmolLM2-135M"
        base_name = input_gguf.stem  # removes .gguf
        # Strip the "-F16" suffix if present so we can append the new type.
        if base_name.endswith("-F16"):
            base_name = base_name[:-4]

        created: dict[str, Path] = {}

        for qtype in quant_types:
            output_path = output_dir / f"{base_name}-{qtype}.gguf"

            logger.info(
                "\n{'=' * 60}\n"
                "  Quantizing: %s → %s\n"
                "  %s\n"
                "{'=' * 60}",
                input_gguf.name, output_path.name, QUANT_TYPES.get(qtype, ""),
            )

            try:
                self.quantize(input_gguf, output_path, qtype)
                created[qtype] = output_path
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.error("❌ Failed to quantize to %s: %s", qtype, exc)
                # Continue with remaining types rather than aborting entirely.

        logger.info(
            "\n🏁 Quantization batch complete — %d / %d variants created",
            len(created), len(quant_types),
        )

        return created
