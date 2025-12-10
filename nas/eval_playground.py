#!/usr/bin/env python3
"""
eval_playground.py

Interactively test code completion models (char-level or token-level).

Usage:
    # Char-level (existing)
    python eval_playground.py --mode char

    # Token-level (BigData)
    python eval_playground.py --mode token \
        --checkpoint logs/train_v1_token_bigdata/v1_token_bigdata_best.pt

    # Batch evaluation
    python eval_playground.py --mode token \
        --checkpoint logs/train_v1_token_bigdata/v1_token_bigdata_best.pt \
        --eval_file eval/prompts/simple_python.txt \
        --output eval/results_token_bigdata.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F

from models import build_model
from datasets import CodeCharVocab
from search_space import ArchitectureConfig

# Token-level support (optional)
try:
    from transformers import GPT2TokenizerFast
    from datasets import CodeTokenVocab
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("[WARN] transformers not available - token mode disabled")


def load_arch_config(model_path: Path) -> tuple:
    """Load architecture config from best model JSON."""
    data = json.loads(model_path.read_text(encoding="utf-8"))

    # Flexible key extraction
    arch = (
        data.get("arch_config")
        or data.get("architecture")
        or data.get("config", {}).get("arch_config")
        or data
    )

    return arch, data


def build_vocab_char(train_path: str = "../data/code_char/train.txt") -> CodeCharVocab:
    """Build character-level vocabulary from training data."""
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(train_path, 'r', encoding='latin-1') as f:
            text = f.read()

    vocab = CodeCharVocab(text)
    return vocab


def build_vocab_token(tokenizer_name: str = "gpt2", tokenizer_path: str = None) -> CodeTokenVocab:
    """Build token-level vocabulary using BPE tokenizer."""
    if not TOKENIZER_AVAILABLE:
        raise RuntimeError("transformers package not installed - cannot use token mode")

    vocab = CodeTokenVocab(tokenizer_name=tokenizer_name, tokenizer_path=tokenizer_path)
    return vocab


def load_model_char(
    model_path: Path,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load char-level model and vocabulary."""
    print(f"[LOAD] Mode: char-level")
    print(f"[LOAD] Loading architecture from {model_path.name}...")

    # Load architecture
    arch, meta = load_arch_config(model_path)

    # Create ArchitectureConfig
    config = ArchitectureConfig(**arch)

    # Build vocabulary
    vocab = build_vocab_char()
    config.vocab_size = vocab.vocab_size

    # Build model
    model = build_model(config)
    model.to(device)

    # Load trained weights if checkpoint provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"[LOAD] Loading trained weights from {checkpoint_path.name}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_loss = checkpoint.get("val_loss", "N/A")
        step = checkpoint.get("step", "N/A")
        print(f"[LOAD] Checkpoint: Step {step}, Val Loss: {val_loss}")
    else:
        print(f"[LOAD] No checkpoint found - using random initialization")

    model.eval()

    print(f"[LOAD] Model: {config.arch_type} L{config.num_layers} H{config.hidden_dim}")
    print(f"[LOAD] Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[LOAD] Device: {device}")
    print(f"[LOAD] Vocab size: {vocab.vocab_size}")
    print()

    return model, vocab, config, device


def load_model_token(
    model_path: Path,
    checkpoint_path: Path,
    tokenizer_name: str = "gpt2",
    tokenizer_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load token-level model and vocabulary."""
    print(f"[LOAD] Mode: token-level (BPE)")
    print(f"[LOAD] Loading architecture from {model_path.name}...")

    # Load architecture
    arch, meta = load_arch_config(model_path)

    # Create ArchitectureConfig
    config = ArchitectureConfig(**arch)

    # Build vocabulary (GPT-2 or custom BPE tokenizer)
    vocab = build_vocab_token(tokenizer_name=tokenizer_name, tokenizer_path=tokenizer_path)
    config.vocab_size = vocab.vocab_size

    # Build model
    model = build_model(config)
    model.to(device)

    # Load trained weights (required for token-level)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Token-level checkpoint not found: {checkpoint_path}")

    print(f"[LOAD] Loading trained weights from {checkpoint_path.name}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss = checkpoint.get("val_loss", "N/A")
    step = checkpoint.get("step", "N/A")
    print(f"[LOAD] Checkpoint: Step {step}, Val Loss: {val_loss}")

    model.eval()

    print(f"[LOAD] Model: {config.arch_type} L{config.num_layers} H{config.hidden_dim}")
    print(f"[LOAD] Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"[LOAD] Device: {device}")
    print(f"[LOAD] Vocab size: {vocab.vocab_size} (tokenizer: {tokenizer_name})")
    print()

    return model, vocab, config, device


@torch.no_grad()
def generate_completion(
    model,
    vocab: Union[CodeCharVocab, 'CodeTokenVocab'],
    prompt: str,
    device: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40
) -> str:
    """
    Generate completion for a given prompt (char or token level).

    Args:
        model: Transformer model
        vocab: Vocabulary (char or token)
        prompt: Input code snippet
        device: Device to run on
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold

    Returns:
        Generated text (continuation only, without prompt)
    """
    # Encode prompt
    input_ids = vocab.encode(prompt)
    if len(input_ids) == 0:
        return "[Empty prompt]"

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Generate tokens
    generated_ids = []

    for _ in range(max_tokens):
        # Forward pass
        logits = model(input_tensor)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')

        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Append to generated sequence
        generated_ids.append(next_token_id)

        # Append to input for next iteration
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

        # Stop if we generate a newline (optional early stopping)
        # Uncomment if you want to stop at newlines:
        # if vocab.itos.get(next_token_id) == '\n':
        #     break

    # Decode generated tokens
    completion = vocab.decode(generated_ids)
    return completion


def run_batch_eval(model, vocab, config, device, args) -> None:
    """Run batch evaluation on prompts from file."""
    import time

    eval_path = Path(args.eval_file)
    out_path = Path(args.output or f"eval/results_{args.mode}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        print(f"Error: Eval file not found: {eval_path}")
        return

    # Read prompts
    prompts = [
        line.rstrip("\n")
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Extract run_id from checkpoint path
    run_id = "unknown"
    if args.checkpoint:
        ckpt_name = Path(args.checkpoint).stem
        # e.g., "v1_token_bigdata_best" -> "token_bigdata_v1"
        if "token_bigdata" in ckpt_name:
            run_id = "token_bigdata_v1"
        elif "token" in ckpt_name:
            run_id = "token_v1"
        elif "production" in ckpt_name:
            run_id = "char_v1_production"
        else:
            run_id = f"{args.mode}_{ckpt_name}"

    print()
    print("=" * 70)
    print(f" Batch Evaluation: {len(prompts)} prompts")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Run ID: {run_id}")
    print(f"Input file: {eval_path}")
    print(f"Output file: {out_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print("=" * 70)
    print()

    results = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, start=1):
            t0 = time.time()
            completion = generate_completion(
                model=model,
                vocab=vocab,
                prompt=prompt,
                device=device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            dt = time.time() - t0

            record = {
                "index": i,
                "prompt": prompt,
                "completion": completion,
                "elapsed_sec": round(dt, 3),
                "mode": args.mode,
                "run_id": run_id,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == len(prompts):
                print(f"[EVAL] {i}/{len(prompts)} done ({dt:.2f}s)")

    print()
    print("=" * 70)
    print(f" Evaluation Complete")
    print("=" * 70)
    print(f"Total prompts: {len(prompts)}")
    print(f"Total time: {sum(r['elapsed_sec'] for r in results):.2f}s")
    print(f"Avg time/prompt: {sum(r['elapsed_sec'] for r in results)/len(prompts):.3f}s")
    print(f"Results saved to: {out_path}")
    print("=" * 70)
    print()


def run_repl(model, vocab, config, device, args) -> None:
    """Run interactive REPL."""
    # Interactive loop
    print("Enter code snippet (or 'quit' to exit):")
    print("=" * 70)
    print()

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip() or prompt.strip().lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break

        # Generate completion
        print()
        print("[Generating...]")
        completion = generate_completion(
            model, vocab, prompt, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        print()
        print("-" * 70)
        print("Prompt:")
        print(prompt)
        print()
        print("Completion:")
        # Handle unicode characters gracefully for Windows console
        try:
            print(completion)
        except UnicodeEncodeError:
            # Fallback: encode with errors='replace' for Windows console
            print(completion.encode(errors='replace').decode())
        print("-" * 70)
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive code completion playground")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["char", "token"],
        default="char",
        help="Model mode: char-level or token-level (default: char)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/codenas_best_current.json",
        help="Path to model JSON file (default: models/codenas_best_current.json)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (required for token mode)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer for token mode (default: gpt2)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer.json file (overrides --tokenizer)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate (default: 64)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8, higher = more random)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="If set, run batch eval mode reading prompts from this file (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file for batch eval results (default: eval/results_{mode}.jsonl)"
    )

    args = parser.parse_args()

    # Validate token mode requirements
    if args.mode == "token":
        if not TOKENIZER_AVAILABLE:
            print("Error: Token mode requires 'transformers' package")
            print("Install with: pip install transformers")
            return

        if not args.checkpoint:
            print("Error: Token mode requires --checkpoint argument")
            print("Example:")
            print("  python eval_playground.py --mode token \\")
            print("    --checkpoint logs/train_v1_token_bigdata/v1_token_bigdata_best.pt")
            return

    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print()
        print("Available models:")
        for p in Path("models").glob("*.json"):
            print(f"  - {p}")
        return

    # Load model
    print("=" * 70)
    print(f" CodeNAS v1 Playground - {args.mode.upper()} Mode")
    print("=" * 70)

    # Default checkpoint for char mode
    if args.mode == "char" and not args.checkpoint:
        args.checkpoint = "logs/train_v1_production/v1_production_best.pt"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    # Load model based on mode
    if args.mode == "char":
        model, vocab, config, device = load_model_char(
            model_path,
            checkpoint_path=checkpoint_path,
            device=args.device
        )
    else:  # token
        model, vocab, config, device = load_model_token(
            model_path,
            checkpoint_path=checkpoint_path,
            tokenizer_name=args.tokenizer,
            tokenizer_path=args.tokenizer_path,
            device=args.device
        )

    # Run in batch eval mode or interactive REPL mode
    if args.eval_file:
        run_batch_eval(model, vocab, config, device, args)
    else:
        run_repl(model, vocab, config, device, args)


if __name__ == "__main__":
    main()
