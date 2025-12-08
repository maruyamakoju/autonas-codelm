#!/usr/bin/env python3
"""
eval_playground.py

Interactively test the current best model on simple code-completion tasks.

Usage:
    cd nas
    python eval_playground.py
    python eval_playground.py --model models/codenas_best_current.json
    python eval_playground.py --max_tokens 64 --temperature 0.8
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from models import build_model
from datasets import CodeCharVocab
from search_space import ArchitectureConfig


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


def build_vocab(train_path: str = "../data/code_char/train.txt") -> CodeCharVocab:
    """Build vocabulary from training data."""
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(train_path, 'r', encoding='latin-1') as f:
            text = f.read()

    vocab = CodeCharVocab(text)
    return vocab


def load_model(
    model_path: Path,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """Load model and vocabulary."""
    print(f"[LOAD] Loading architecture from {model_path.name}...")

    # Load architecture
    arch, meta = load_arch_config(model_path)

    # Create ArchitectureConfig
    config = ArchitectureConfig(**arch)

    # Build vocabulary
    vocab = build_vocab()
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


@torch.no_grad()
def generate_completion(
    model,
    vocab: CodeCharVocab,
    prompt: str,
    device: str,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40
) -> str:
    """
    Generate completion for a given prompt.

    Args:
        model: Transformer model
        vocab: Vocabulary
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive code completion playground")
    parser.add_argument(
        "--model",
        type=str,
        default="models/codenas_best_current.json",
        help="Path to model JSON file (default: models/codenas_best_current.json)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/train_v1_production/v1_production_best.pt",
        help="Path to trained checkpoint (default: logs/train_v1_production/v1_production_best.pt)"
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

    args = parser.parse_args()

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
    print(" CodeNAS v1 Playground")
    print("=" * 70)

    # Check checkpoint path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    model, vocab, config, device = load_model(
        model_path,
        checkpoint_path=checkpoint_path,
        device=args.device
    )

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


if __name__ == "__main__":
    main()
