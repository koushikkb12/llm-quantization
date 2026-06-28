#!/usr/bin/env python3
"""
Step 8: Upload quantized models to HuggingFace Hub.

=== Sharing your work ===
The whole point of quantization is to make models accessible! This script:
  1. Creates a HuggingFace repository for your quantized models
  2. Generates a detailed model card with benchmark results
  3. Uploads all GGUF files (splitting large files if needed)
  4. Makes your quants available to the community

=== Model Card Best Practices ===
A good model card should include:
  - Original model link and description
  - Quantization method and settings
  - Perplexity benchmarks (original vs quantized)
  - RAM/VRAM requirements per quant type
  - Recommended quant for different hardware tiers
  - Whether imatrix calibration was used
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantizer import GGUFQuantizer
from src.utils import get_model_size

def generate_model_card(args, models_info) -> str:
    """Generate a README.md for the HuggingFace repo."""
    content = [
        "---",
        "tags:",
        "- quantized",
        "- gguf",
        "- llama-cpp",
        "---",
        f"# Quantized Models for {args.original_model or 'Original Model'}\n",
        f"These are GGUF format quantized models for [{args.original_model or 'the original model'}](https://huggingface.co/{args.original_model}).\n",
        "## Quantization Info\n",
        f"- **Method:** llama.cpp",
        f"- **iMatrix Calibration:** {'Yes' if args.imatrix_used else 'No'}\n",
        "## Available Files\n",
        "| File | Size | Note |",
        "|---|---|---|"
    ]
    
    for info in sorted(models_info, key=lambda x: x["size_mb"]):
        note = ""
        name = info['name'].upper()
        if "Q4_K_M" in name:
            note = "Recommended for most users"
        elif "Q8" in name:
            note = "Highest quality, largest size"
        elif "Q2" in name:
            note = "Maximum compression"
            
        content.append(f"| {info['name']} | {info['size_mb']:.1f} MB | {note} |")
        
    content.extend([
        "\n## RAM/VRAM Requirements\n",
        "As a rule of thumb, you need at least 1.2x the file size in RAM/VRAM to load the model.\n",
        "## Usage\n",
        "These models can be loaded with standard GGUF tools like `llama.cpp`, `ollama`, or `LM Studio`."
    ])
    
    return "\n".join(content)

def main():
    parser = argparse.ArgumentParser(description="Upload models to HuggingFace Hub.")
    parser.add_argument("--models-dir", required=True, help="Directory containing .gguf files")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g. username/Model-GGUF)")
    parser.add_argument("--original-model", default="", help="Original model ID for model card")
    parser.add_argument("--benchmark-results", help="Path to benchmark JSON")
    parser.add_argument("--perplexity-results", help="Path to perplexity JSON")
    parser.add_argument("--imatrix-used", action="store_true", help="Flag if imatrix was used")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--split-size", type=float, default=49.0, help="Max GB before splitting (default: 49.0)")
    
    args = parser.parse_args()
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
        
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Error: Models dir not found: {models_dir}")
        sys.exit(1)
        
    gguf_files = list(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"❌ No .gguf files found in {models_dir}")
        sys.exit(1)
        
    api = HfApi()
    
    print(f"\n🚀 Preparing to upload to {args.repo_id}...")
    
    # 1. Split large files
    quantizer = GGUFQuantizer()
    max_bytes = args.split_size * 1024 * 1024 * 1024
    
    models_info = []
    files_to_upload = []
    
    for f in gguf_files:
        size = f.stat().st_size
        if size > max_bytes:
            print(f"✂️  Splitting large file {f.name}...")
            split_files = quantizer.split_gguf(f, models_dir, args.split_size)
            files_to_upload.extend(split_files)
        else:
            files_to_upload.append(f)
            
        models_info.append({
            "name": f.name,
            "size_mb": size / (1024 * 1024)
        })
        
    # 2. Create repo
    print(f"📦 Creating repo {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    
    # 3. Generate and save README.md if it doesn't exist
    readme_path = models_dir / "README.md"
    if not readme_path.exists():
        print("📝 Generating default README.md...")
        with open(readme_path, "w") as f:
            f.write(generate_model_card(args, models_info))
    else:
        print("📄 Using existing README.md...")
    
    # 4. Upload
    print(f"📤 Uploading files...")
    # This automatically shows a progress bar via huggingface_hub
    api.upload_folder(
        folder_path=str(models_dir),
        repo_id=args.repo_id,
        repo_type="model"
    )
    
    print(f"\n✅ Upload complete! View your model at:")
    print(f"   https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
