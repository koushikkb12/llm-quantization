#!/usr/bin/env python3
"""
Step 5: Interactive chat with a quantized model.

This script provides a beautiful terminal chat interface using the
Rich library. Load any quantized GGUF model and have a conversation!

Features:
  - Streaming text generation (see tokens appear in real-time)
  - Token speed display after each response
  - Model info and stats commands
  - Beautiful Rich-powered UI

=== This is the payoff! ===
After downloading, converting, and quantizing, this is where you
actually USE the quantized model. Try chatting with different
quantization levels and notice:
  - Speed differences (Q2_K is fastest, Q8_0 is slowest)
  - Quality differences (Q8_0 produces best text, Q2_K may be garbled)
  - The sweet spot is usually Q4_K_M or Q5_K_M
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import get_models_dir, get_model_size, format_size


def create_chat(model_path: str, context_size: int, temperature: float):
    """
    Launch an interactive chat session with a quantized model.
    
    Args:
        model_path: Path to the .gguf model file
        context_size: Maximum context window size in tokens
        temperature: Sampling temperature (higher = more creative)
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text
    from llama_cpp import Llama
    
    console = Console()
    model_path = Path(model_path)
    
    # Check model exists
    if not model_path.exists():
        console.print(f"[red]❌ Model not found: {model_path}[/red]")
        console.print("[yellow]Run scripts 01-03 first to create quantized models.[/yellow]")
        return
    
    # Load model
    console.print(f"\n[cyan]⏳ Loading model: {model_path.name}...[/cyan]")
    load_start = time.perf_counter()
    
    llm = Llama(
        model_path=str(model_path),
        n_ctx=context_size,
        n_threads=8,
        n_gpu_layers=-1,  # Offload ALL layers to GPU for max speed
        verbose=False,
    )
    
    load_time = time.perf_counter() - load_start
    model_size = get_model_size(model_path)
    
    # Detect GPU
    try:
        import subprocess as _sp
        _gpu_result = _sp.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu_name = _gpu_result.stdout.strip() if _gpu_result.returncode == 0 else "None (CPU only)"
    except (FileNotFoundError, _sp.TimeoutExpired):
        gpu_name = "None (CPU only)"
    
    # Welcome banner
    welcome = f"""[bold cyan]🧪 LLM Quantization Lab — Chat Interface[/bold cyan]

[dim]Model:    {model_path.name}
Size:     {model_size:.1f} MB
GPU:      {gpu_name}
Context:  {context_size} tokens
Temp:     {temperature}
Loaded:   {load_time:.1f}s[/dim]

[dim]Commands: /help, /stats, /clear, /quit[/dim]"""
    
    console.print(Panel(welcome, border_style="cyan", padding=(1, 2)))
    
    # Chat history for context
    messages = []
    total_tokens = 0
    total_time = 0.0
    num_responses = 0
    
    while True:
        try:
            # Get user input
            console.print()
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd == "/quit" or cmd == "/exit":
                    console.print("\n[dim]👋 Goodbye![/dim]")
                    break
                
                elif cmd == "/help":
                    help_text = """[bold]Available Commands:[/bold]
  /help   — Show this help message
  /stats  — Show session statistics
  /clear  — Clear chat history
  /quit   — Exit the chat"""
                    console.print(Panel(help_text, title="Help", border_style="blue"))
                    continue
                
                elif cmd == "/stats":
                    avg_speed = total_tokens / total_time if total_time > 0 else 0
                    stats_text = f"""[bold]Session Statistics:[/bold]
  Model:          {model_path.name}
  Model size:     {model_size:.1f} MB
  Load time:      {load_time:.1f}s
  Messages:       {num_responses}
  Total tokens:   {total_tokens}
  Avg speed:      {avg_speed:.1f} tok/s
  Context used:   ~{sum(len(m.get('content', '')) for m in messages)} chars"""
                    console.print(Panel(stats_text, title="Stats", border_style="yellow"))
                    continue
                
                elif cmd == "/clear":
                    messages.clear()
                    console.clear()
                    console.print("[dim]🧹 Chat history cleared.[/dim]")
                    continue
                
                else:
                    console.print(f"[red]Unknown command: {cmd}. Type /help for options.[/red]")
                    continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            console.print("[bold cyan]AI:[/bold cyan] ", end="")
            
            gen_start = time.perf_counter()
            token_count = 0
            response_text = ""
            
            # Use chat completion with streaming
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=512,
                    temperature=temperature,
                    stream=True,
                )
                
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        console.print(content, end="", highlight=False)
                        response_text += content
                        token_count += 1
                        
            except Exception as e:
                # Fallback to simple completion if chat format isn't supported
                prompt_text = user_input
                output = llm(
                    prompt_text,
                    max_tokens=512,
                    temperature=temperature,
                    echo=False,
                )
                response_text = output["choices"][0]["text"]
                token_count = output["usage"]["completion_tokens"]
                console.print(response_text, highlight=False)
            
            gen_time = time.perf_counter() - gen_start
            tokens_per_sec = token_count / gen_time if gen_time > 0 else 0
            
            # Update stats
            total_tokens += token_count
            total_time += gen_time
            num_responses += 1
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response_text})
            
            # Show speed
            console.print(f"\n[dim]  [{token_count} tokens, {tokens_per_sec:.1f} tok/s, {gen_time:.1f}s][/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[dim]👋 Interrupted. Goodbye![/dim]")
            break
        except EOFError:
            console.print("\n[dim]👋 Goodbye![/dim]")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with a quantized GGUF model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_chat.py                                        # Use default Q4_K_M
  python 05_chat.py --model models/quantized/model-Q8_0.gguf  # Higher quality
  python 05_chat.py --model models/quantized/model-Q2_K.gguf  # Faster, lower quality
  python 05_chat.py --temperature 0.3                          # More focused responses
  python 05_chat.py --temperature 1.2                          # More creative responses
        """
    )
    
    dirs = get_models_dir()
    default_model = dirs["quantized"] / "model-Q4_K_M.gguf"
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model),
        help=f"Path to .gguf model file (default: {default_model})"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=2048,
        help="Maximum context window size in tokens (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature: 0.0=deterministic, 1.0+=creative (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    create_chat(args.model, args.context_size, args.temperature)


if __name__ == "__main__":
    main()
