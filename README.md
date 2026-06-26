# LLM Quantization Lab

> **Shrink any Large Language Model by up to 8x and run it locally — learn quantization hands-on.**

A complete, end-to-end pipeline for downloading, converting, quantizing, benchmarking, and chatting with open-source LLMs using [llama.cpp](https://github.com/ggerganov/llama.cpp). Clone this repo, pick a model, and compress it from full FP16 precision all the way down to 2-bit — then see exactly how much speed you gain and how much quality you lose.

---

## Features

- **Download** any HuggingFace model with a single command
- **Convert** HuggingFace safetensors to GGUF format (F16 baseline)
- **Quantize** to 5 precision levels: Q8_0, Q5_K_M, Q4_K_M, Q4_0, Q2_K
- **Benchmark** all variants side-by-side (tokens/sec, load time, model size, RAM usage)
- **Chat** interactively with any quantized model
- **GPU Accelerated** — auto-detects NVIDIA GPUs and offloads all layers to VRAM

---

## System Requirements

Quantization is a **CPU + RAM intensive** process. The GPU is used for *inference* (running the model), not for the quantization step itself.

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|------------|-------|
| **OS** | Linux (Ubuntu 20.04+) or macOS | Windows via WSL2 works too |
| **Python** | 3.10+ | Tested with 3.12 |
| **CMake** | 3.14+ | For building llama.cpp |
| **C++ Compiler** | GCC 9+ or Clang 12+ | For building llama.cpp |
| **Git** | 2.0+ | For cloning llama.cpp |

### RAM Requirements (the most important factor)

The **conversion step** (HuggingFace to GGUF) loads the entire model into CPU RAM. You need **at least 2x the model's FP16 size** in available RAM:

| Model Size (Parameters) | FP16 Weight Size | Minimum RAM Required | Recommended RAM |
|--------------------------|-----------------|---------------------|-----------------|
| **135M** (SmolLM2-135M) | ~270 MB | 1 GB | 2 GB |
| **1.5B** (Qwen2.5-1.5B) | ~3 GB | 6 GB | 8 GB |
| **3B** (Qwen2.5-3B) | ~6 GB | 12 GB | 16 GB |
| **7B** (Llama-3-8B) | ~14 GB | 28 GB | 32 GB |
| **13B** (Llama-2-13B) | ~26 GB | 52 GB | 64 GB |
| **70B** (Llama-3-70B) | ~140 GB | 280 GB | 320 GB |

> [!IMPORTANT]
> **You do NOT need a GPU to quantize models.** Quantization runs entirely on CPU + RAM. A GPU is only needed to *run* (infer) the quantized models faster. Without a GPU, the models still work — just slower (CPU inference).

### GPU Requirements (for inference only)

To run the quantized models with GPU acceleration, you need an NVIDIA GPU with enough VRAM to hold the **quantized** model (not the original):

| Quantized Model | Q4_K_M Size | GPU VRAM Needed | Example GPUs |
|-----------------|-------------|-----------------|--------------|
| **135M** | ~100 MB | Any GPU | GTX 1060, T4, etc. |
| **3B** | ~1.8 GB | 4+ GB | RTX 3060, T4, L4 |
| **7B** | ~4.4 GB | 6+ GB | RTX 3060, RTX 4060, L4 |
| **13B** | ~7.9 GB | 10+ GB | RTX 3080, RTX 4070, A10 |
| **70B** | ~40 GB | 48+ GB | A100 (80GB), 2x RTX 4090 |

> [!TIP]
> **Sweet spot for most users:** A machine with **16-32 GB RAM** and a **GPU with 8+ GB VRAM** can handle models up to ~7B parameters comfortably. This covers the vast majority of useful open-source models.

---

## Project Structure

```
llm-quantization-lab/
├── README.md                      # You are here
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes models, builds, caches
│
├── scripts/                       # Step-by-step pipeline scripts
│   ├── 01_download_model.py       # Download any HuggingFace model
│   ├── 02_convert_to_gguf.py      # Convert to GGUF format (F16)
│   ├── 03_quantize.py             # Quantize to Q8, Q5, Q4, Q2
│   ├── 04_benchmark.py            # Benchmark all variants
│   └── 05_chat.py                 # Interactive chat interface
│
├── src/                           # Core library modules
│   ├── __init__.py
│   ├── quantizer.py               # llama.cpp build & quantize wrapper
│   ├── benchmarker.py             # Benchmarking engine
│   └── utils.py                   # Shared constants & helpers
│
├── notebooks/
│   └── quantization_explainer.md  # Deep-dive guide on how quantization works
│
├── models/                        # Auto-created, git-ignored
│   ├── original/                  # Downloaded HuggingFace weights
│   └── quantized/                 # Generated GGUF files
│
├── results/                       # Benchmark outputs
│   └── .gitkeep
│
└── llama.cpp/                     # Auto-cloned, git-ignored
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/koushikkb12/llm-quantization.git
cd llm-quantization
```

### 2. Install dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

> [!NOTE]
> **GPU users:** If you have an NVIDIA GPU and want GPU-accelerated inference, install `llama-cpp-python` with CUDA support:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
> ```

### 3. Run the pipeline

Each script is numbered and designed to be run in order. Each one prints educational information about what it's doing and why.

```bash
# Step 1: Download a model from HuggingFace
python scripts/01_download_model.py
# Default: SmolLM2-135M (~270MB). For a bigger model:
# python scripts/01_download_model.py --model-id Qwen/Qwen2.5-3B-Instruct

# Step 2: Convert HuggingFace weights to GGUF format (F16 baseline)
python scripts/02_convert_to_gguf.py
# For a custom model:
# python scripts/02_convert_to_gguf.py --model-dir models/original/Qwen2.5-3B-Instruct --output models/quantized/Qwen2.5-3B-F16.gguf

# Step 3: Quantize to different bit-widths (Q8, Q5, Q4, Q2)
python scripts/03_quantize.py
# Specific types only:
# python scripts/03_quantize.py --types Q4_K_M Q8_0

# Step 4: Benchmark all variants
python scripts/04_benchmark.py

# Step 5: Chat with your favourite model
python scripts/05_chat.py --model models/quantized/model-Q4_K_M.gguf
```

---

## Example Benchmark Results

Here are real results from quantizing **Qwen2.5-3B-Instruct** on an L4 GPU:

### Size Comparison

| Quantization | Size | Compression | Size Saved |
|-------------|------|-------------|------------|
| **F16** (baseline) | 5,892 MB | 1.0x | — |
| **Q8_0** | 3,133 MB | 1.9x | 47% |
| **Q5_K_M** | 2,122 MB | 2.8x | 64% |
| **Q4_K_M** | 1,841 MB | 3.2x | 69% |
| **Q4_0** | 1,738 MB | 3.4x | 70% |
| **Q2_K** | 1,216 MB | 4.8x | 79% |

### Speed Comparison (L4 GPU, 100 tokens generated)

| Quantization | Tokens/sec | Load Time | Total Time |
|-------------|-----------|-----------|------------|
| **F16** | 36.9 | 2.51s | 4.75s |
| **Q8_0** | 64.4 | 1.42s | 2.87s |
| **Q5_K_M** | 86.4 | 1.15s | 2.31s |
| **Q4_K_M** | 98.8 | 1.03s | 2.04s |
| **Q4_0** | 105.1 | 1.04s | 1.99s |
| **Q2_K** | 112.7 | 0.85s | 1.22s |

> **Key insight:** Q4_K_M delivers **2.7x the speed** of F16 while using **69% less disk space** — with minimal quality loss. This is why it's the most popular quantization level for deployment.

---

## Understanding Quantization

### What is Quantization?

Quantization compresses neural network weights by reducing the numerical precision of each parameter. Instead of storing each weight as a 16-bit floating-point number (FP16), we map them to lower-bit integers.

```
FP16:  Each weight uses 16 bits  ->  2 bytes per parameter
Q8_0:  Each weight uses  8 bits  ->  1 byte  per parameter  (2x smaller)
Q4_K_M: Each weight uses ~4 bits ->  0.5 bytes per parameter (4x smaller)
Q2_K:  Each weight uses ~2 bits  ->  0.25 bytes per parameter (8x smaller)
```

### How does it work?

For each block of weights (typically 32 values):

1. Find the **min** and **max** values in the block
2. Calculate a **scale factor**: `scale = (max - min) / (2^bits - 1)`
3. Map each float to the nearest integer: `q = round((weight - min) / scale)`
4. Store: `[scale, zero_point, quantized_values...]`

### K-Quant (the smart method)

The `K` in types like `Q4_K_M` stands for **k-quant**, a smarter approach that assigns different precision to different layers:

- **Attention layers** get higher precision (these are more sensitive to errors)
- **Feed-forward layers** get lower precision (these are more tolerant)

The suffix indicates the aggressiveness:
- `_S` = Small — more aggressive compression
- `_M` = Medium — balanced (recommended)
- `_L` = Large — less compression, higher quality

This is why `Q4_K_M` generally outperforms `Q4_0` despite having similar file sizes.

### Which quantization level should I use?

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Maximum quality** | Q8_0 | Nearly indistinguishable from FP16 |
| **Best balance** | Q4_K_M | Most popular — great quality at 4x compression |
| **Speed priority** | Q4_0 | Slightly faster than Q4_K_M |
| **Minimum size** | Q2_K | For very constrained environments |
| **Experimentation** | Run all 5 | That's what this lab is for |

> For a deep dive, see [notebooks/quantization_explainer.md](notebooks/quantization_explainer.md).

---

## Advanced Usage

### Using a custom model

You can quantize **any** HuggingFace model that uses safetensors:

```bash
# Download any model
python scripts/01_download_model.py --model-id mistralai/Mistral-7B-Instruct-v0.3

# Convert (point to the downloaded directory)
python scripts/02_convert_to_gguf.py \
  --model-dir models/original/Mistral-7B-Instruct-v0.3 \
  --output models/quantized/Mistral-7B-F16.gguf

# Quantize
python scripts/03_quantize.py --input models/quantized/Mistral-7B-F16.gguf

# Benchmark
python scripts/04_benchmark.py

# Chat
python scripts/05_chat.py --model models/quantized/Mistral-7B-Q4_K_M.gguf
```

### Quantize only specific types

```bash
# Only Q4_K_M and Q8_0
python scripts/03_quantize.py --types Q4_K_M Q8_0
```

### Chat with custom settings

```bash
# Adjust temperature and max tokens
python scripts/05_chat.py \
  --model models/quantized/model-Q4_K_M.gguf \
  --temperature 0.7 \
  --max-tokens 512
```

---

## How the Pipeline Works

```
+----------------+     +----------------+     +----------------+
|  HuggingFace   |     |   F16 GGUF     |     |   Quantized    |
|  Safetensors   |---->|   Baseline     |---->|  Q8/Q5/Q4/Q2   |
|   (~6 GB)      |     |   (~6 GB)      |     |  (~1-3 GB)     |
+----------------+     +----------------+     +----------------+
  01_download          02_convert_to_gguf       03_quantize
      |                                             |
 CPU + Internet                                CPU + RAM only
                                                    |
                    +----------------+     +----------------+
                    |   Chat UI      |<----|   Benchmark    |
                    |   (GPU)        |     |   (GPU)        |
                    +----------------+     +----------------+
                       05_chat              04_benchmark
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — The C/C++ inference engine that makes local LLMs possible
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings for llama.cpp
- [HuggingFace](https://huggingface.co/) — The home of open-source AI models
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — The tiny model that's perfect for learning
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — Punches way above its weight class
