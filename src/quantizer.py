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
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .utils import QUANT_TYPES, get_project_root

logger = logging.getLogger(__name__)


def _safe_run(cmd, *args, **kwargs):
    """Run a subprocess through shell to reset FPU trapping state and prevent SIGFPE."""
    import shlex
    if isinstance(cmd, list):
        cmd_str = shlex.join(cmd)
    else:
        cmd_str = cmd
    
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        if hasattr(libc, 'fedisableexcept'):
            libc.fedisableexcept(0x3f)  # FE_ALL_EXCEPT is 0x3f on x86_64
    except Exception as e:
        logger.warning("⚠️ Could not disable FPU exceptions: %s", e)
        
    kwargs["shell"] = True
    return subprocess.run(cmd_str, *args, **kwargs)


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

        # The imatrix binary generates importance matrices for calibrated quantization.
        self.imatrix_binary: Path = self.llama_cpp_dir / "build" / "bin" / "llama-imatrix"

        # The perplexity binary measures model quality.
        self.perplexity_binary: Path = self.llama_cpp_dir / "build" / "bin" / "llama-perplexity"

        # The gguf-split binary splits large GGUF files for HuggingFace upload.
        self.gguf_split_binary: Path = self.llama_cpp_dir / "build" / "bin" / "llama-gguf-split"

        # Disable floating-point exception trapping to prevent subprocess crashes (like SIGFPE).
        # This is needed because some Python packages (like numpy/torch) enable exception trapping
        # which subprocesses inherit, causing llama.cpp binaries to crash on division by zero / log(0).
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            if hasattr(libc, 'fedisableexcept'):
                libc.fedisableexcept(0x3f)  # FE_ALL_EXCEPT is 0x3f on x86_64
        except Exception as e:
            logger.warning("⚠️ Could not disable FPU exceptions: %s", e)

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
    # Calibration & iMatrix
    # ──────────────────────────────────────────────────────────────────────

    def _get_default_calibration_data(self) -> Path:
        """Download and prepare default calibration data for imatrix generation.
        
        Uses the WikiText-2 test split as calibration data. This is the standard
        calibration dataset used by the quantization community because it contains
        diverse, high-quality English text that exercises a wide range of the
        model's capabilities.
        
        Returns:
            Path: Path to the calibration text file.
        """
        calibration_dir = get_project_root() / "models" / "calibration"
        calibration_file = calibration_dir / "calibration_data.txt"
        
        if calibration_file.exists():
            logger.info("📄 Using existing calibration data: %s", calibration_file)
            return calibration_file
        
        calibration_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📥 Downloading WikiText-2 calibration data...")
        loaded = False
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            
            with open(calibration_file, "w", encoding="utf-8") as f:
                for i, item in enumerate(dataset):
                    text = item["text"].strip()
                    if text:  # Skip empty lines
                        f.write(text + "\n")
                    if i >= 500:  # Use first ~500 non-empty entries
                        break
            
            logger.info("✅ Calibration data saved to %s", calibration_file)
            loaded = True
        except Exception as e:
            logger.warning("⚠️  Could not load dataset via datasets library: %s. Trying manual download...", e)
            
        if not loaded:
            try:
                import urllib.request
                import zipfile
                import io
                
                zip_url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
                logger.info("📥 Downloading WikiText-2 zip from %s...", zip_url)
                
                req = urllib.request.Request(
                    zip_url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response:
                    zip_data = response.read()
                
                with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                    test_file_name = [name for name in z.namelist() if "wiki.test.raw" in name][0]
                    with z.open(test_file_name) as f_in:
                        raw_text = f_in.read().decode('utf-8')
                
                lines = raw_text.splitlines()
                count = 0
                with open(calibration_file, "w", encoding="utf-8") as f_out:
                    for line in lines:
                        text = line.strip()
                        if text:
                            f_out.write(text + "\n")
                            count += 1
                            if count >= 500:
                                break
                logger.info("✅ Calibration data saved to %s (manually downloaded)", calibration_file)
                loaded = True
            except Exception as e_manual:
                logger.error("❌ Manual calibration data download failed: %s", e_manual)
        
        if not loaded:
            # Fallback: create a minimal but sufficiently long calibration file (>1024 tokens)
            with open(calibration_file, "w", encoding="utf-8") as f:
                fallback_sentences = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Artificial intelligence is transforming the world of technology.",
                    "Large language models can generate coherent text across many domains.",
                    "Quantization reduces model size while preserving output quality.",
                    "The transformer architecture revolutionized natural language processing."
                ]
                repeated_text = " ".join(fallback_sentences * 100) + "\n"
                f.write(repeated_text)
            logger.warning("⚠️  Using repeated fallback calibration data (length: %d chars)", len(repeated_text))
        return calibration_file

    def generate_imatrix(
        self,
        model_gguf: Path,
        output_path: Path,
        calibration_file: Optional[Path] = None,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        fit: str = "on",
        chunks: int = -1,
    ) -> Path:
        """Generate an importance matrix (imatrix) for calibrated quantization.
        
        An importance matrix measures how much each weight contributes to the
        model's output quality. This information is then used during quantization
        to allocate more bits to important weights and fewer bits to less
        important ones.
        
        **Why imatrix matters for MoE models:**
        
        Standard k-quant heuristics assign precision based on layer *type*
        (attention vs feed-forward). But in Mixture-of-Experts models, different
        experts within the same layer can have wildly different importance.
        An imatrix captures this per-weight importance, leading to dramatically
        better quality for MoE quantization.
        
        **The process:**
        1. Feed calibration text through the model
        2. Track the magnitude of each weight's contribution
        3. Save importance scores to a .dat file
        4. Pass .dat to llama-quantize for importance-aware quantization
        
        Args:
            model_gguf: Path to the GGUF model file (typically F16 baseline).
            output_path: Where to save the importance matrix .dat file.
            calibration_file: Path to calibration text file. If None, downloads
                             WikiText-2 automatically.
            n_ctx: Context window size for calibration (default: 512).
            n_batch: Batch size for processing (default: 512).
            n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only).
        
        Returns:
            Path: Path to the generated importance matrix file.
        
        Raises:
            FileNotFoundError: If model or imatrix binary not found.
            subprocess.CalledProcessError: If imatrix generation fails.
        """
        if not model_gguf.exists():
            raise FileNotFoundError(f"Model GGUF not found: {model_gguf}")
        
        if not self.imatrix_binary.exists():
            raise FileNotFoundError(
                f"imatrix binary not found: {self.imatrix_binary}. "
                f"Rebuild llama.cpp with clone_and_build()."
            )
        
        if calibration_file is None:
            calibration_file = self._get_default_calibration_data()
        
        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "🔬 Generating importance matrix...\n"
            "   Model       : %s\n"
            "   Calibration : %s\n"
            "   Output      : %s\n"
            "   Context     : %d tokens\n"
            "   GPU layers  : %s\n"
            "   Fit mode    : %s\n"
            "   Chunks      : %s",
            model_gguf.name, calibration_file.name, output_path,
            n_ctx, "ALL" if n_gpu_layers == -1 else str(n_gpu_layers), fit,
            "ALL" if chunks == -1 else str(chunks),
        )
        
        cmd = [
            str(self.imatrix_binary),
            "-m", str(model_gguf),
            "-f", str(calibration_file),
            "-o", str(output_path),
            "--ctx-size", str(n_ctx),
            "-b", str(n_batch),
            "-ngl", str(n_gpu_layers),
            "--output-frequency", "10",  # Write every 10 iterations to avoid division-by-zero crash in llama.cpp
            "--fit", fit,
        ]
        if chunks > 0:
            cmd.extend(["--chunks", str(chunks)])
        
        _safe_run(cmd, check=True)
        
        logger.info("✅ Importance matrix generated → %s", output_path)
        return output_path

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
        _safe_run(
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

    def quantize(self, input_gguf: Path, output_gguf: Path, quant_type: str, imatrix_path: Optional[Path] = None) -> Path:
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
        # Usage: llama-quantize [--imatrix <imatrix.dat>] <input.gguf> <output.gguf> <type>
        cmd = [str(self.quantize_binary)]
        if imatrix_path is not None:
            if not imatrix_path.exists():
                raise FileNotFoundError(f"Importance matrix not found: {imatrix_path}")
            cmd.extend(["--imatrix", str(imatrix_path)])
        
        cmd.extend([
            str(input_gguf),
            str(output_gguf),
            quant_type,
        ])
        
        _safe_run(cmd, check=True)

        logger.info("✅ Quantization complete → %s", output_gguf)
        return output_gguf

    def quantize_all(
        self,
        input_gguf: Path,
        output_dir: Path,
        quant_types: Optional[List[str]] = None,
        imatrix_path: Optional[Path] = None,
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
                self.quantize(input_gguf, output_path, qtype, imatrix_path=imatrix_path)
                created[qtype] = output_path
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                logger.error("❌ Failed to quantize to %s: %s", qtype, exc)
                # Continue with remaining types rather than aborting entirely.

        logger.info(
            "\n🏁 Quantization batch complete — %d / %d variants created",
            len(created), len(quant_types),
        )

        return created

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation & Utilities
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_perplexity(
        self,
        model_gguf: Path,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
    ) -> dict:
        """Evaluate model perplexity using llama-perplexity.
        
        Perplexity is the standard metric for measuring language model quality.
        Lower values indicate better quality. A perplexity of N means the model
        is, on average, as uncertain as choosing uniformly among N options.
        
        Typical perplexity ranges for quantized models:
            - F16 baseline: ~5-8 (depending on model)
            - Q8_0: +0.01-0.05 above baseline
            - Q4_K_M: +0.05-0.2 above baseline
            - Q2_K: +0.5-5.0 above baseline (significant degradation)
        
        Args:
            model_gguf: Path to the GGUF model file to evaluate.
            n_ctx: Context window size for evaluation.
            n_batch: Batch size for processing.
            n_gpu_layers: GPU layers to offload (-1 = all).
        
        Returns:
            dict: Results containing 'model_name', 'perplexity', 'n_ctx'.
        
        Raises:
            FileNotFoundError: If model or perplexity binary not found.
            RuntimeError: If perplexity cannot be parsed from output.
        """
        if not model_gguf.exists():
            raise FileNotFoundError(f"Model GGUF not found: {model_gguf}")
        
        if not self.perplexity_binary.exists():
            raise FileNotFoundError(
                f"Perplexity binary not found: {self.perplexity_binary}. "
                f"Rebuild llama.cpp with clone_and_build()."
            )
        
        calibration_file = self._get_default_calibration_data()
        
        logger.info(
            "📏 Evaluating perplexity...\n"
            "   Model   : %s\n"
            "   Context : %d tokens",
            model_gguf.name, n_ctx,
        )
        
        cmd = [
            str(self.perplexity_binary),
            "-m", str(model_gguf),
            "-f", str(calibration_file),
            "--ctx-size", str(n_ctx),
            "-b", str(n_batch),
            "-ngl", str(n_gpu_layers),
        ]
        
        result = _safe_run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        
        # Parse perplexity from output.
        # llama-perplexity outputs lines like:
        # "Final estimate: PPL = 5.1234 +/- 0.0567"
        ppl_value = None
        for line in output.split("\n"):
            # Try multiple patterns that llama-perplexity might output
            match = re.search(r'(?:Final estimate:|perplexity).*?=\s*([\d.]+)', line, re.IGNORECASE)
            if match:
                ppl_value = float(match.group(1))
        
        if ppl_value is None:
            # Fallback: look for any line with "PPL" and a number
            for line in output.split("\n"):
                match = re.search(r'PPL\s*=?\s*([\d.]+)', line)
                if match:
                    ppl_value = float(match.group(1))
        
        if ppl_value is None:
            logger.error("Could not parse perplexity from output:\n%s", output[-2000:])
            raise RuntimeError(
                "Failed to parse perplexity value from llama-perplexity output. "
                "Check the model file and calibration data."
            )
        
        result_dict = {
            "model_name": model_gguf.name,
            "perplexity": round(ppl_value, 4),
            "n_ctx": n_ctx,
        }
        
        logger.info(
            "✅ Perplexity: %.4f (model: %s)",
            ppl_value, model_gguf.name,
        )
        
        return result_dict

    def split_gguf(
        self,
        input_gguf: Path,
        output_dir: Path,
        max_size_gb: float = 49.0,
    ) -> List[Path]:
        """Split a large GGUF file into smaller parts for upload.
        
        HuggingFace Hub has a 50 GB file size limit. Models larger than this
        need to be split into multiple parts. The llama-gguf-split tool handles
        this while preserving the GGUF format's integrity.
        
        Users can load split models by pointing their GGUF loader at the first
        part file — the loader will automatically find and load subsequent parts.
        
        Args:
            input_gguf: Path to the GGUF file to split.
            output_dir: Directory to write split files into.
            max_size_gb: Maximum size per split file in GB (default: 49.0,
                         just under HuggingFace's 50 GB limit).
        
        Returns:
            List[Path]: Paths to all created split files.
        
        Raises:
            FileNotFoundError: If input file or split binary not found.
            subprocess.CalledProcessError: If splitting fails.
        """
        if not input_gguf.exists():
            raise FileNotFoundError(f"Input GGUF not found: {input_gguf}")
        
        if not self.gguf_split_binary.exists():
            raise FileNotFoundError(
                f"gguf-split binary not found: {self.gguf_split_binary}. "
                f"Rebuild llama.cpp with clone_and_build()."
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output basename for split files
        output_base = output_dir / input_gguf.stem
        
        logger.info(
            "✂️  Splitting GGUF file...\n"
            "   Input    : %s\n"
            "   Max size : %.1f GB\n"
            "   Output   : %s",
            input_gguf.name, max_size_gb, output_dir,
        )
        
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        _safe_run(
            [
                str(self.gguf_split_binary),
                "--split",
                "--split-max-size", str(max_size_bytes),
                str(input_gguf),
                str(output_base),
            ],
            check=True,
        )
        
        # Find all created split files
        split_files = sorted(output_dir.glob(f"{input_gguf.stem}-*"))
        
        logger.info(
            "✅ Split into %d files in %s",
            len(split_files), output_dir,
        )
        
        return split_files
