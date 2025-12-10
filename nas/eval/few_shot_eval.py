"""
Few-shot evaluation for CodeLM

Key insight: Small language models need context to avoid mode collapse.
This script evaluates generation quality using proper few-shot prompts.
"""

import torch
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from search_space import ArchitectureConfig
from models import build_model
from datasets import CodeTokenVocab


# Few-shot prompt templates
FEW_SHOT_TEMPLATES = {
    "function": {
        "prefix": '''# Python function examples

def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

''',
        "prompts": [
            "def subtract(a, b):",
            "def divide(a, b):",
            "def max_value(a, b):",
            "def factorial(n):",
            "def is_even(n):",
        ]
    },
    "class": {
        "prefix": '''# Python class examples

class Counter:
    """A simple counter class."""

    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

''',
        "prompts": [
            "class Timer:",
            "class Stack:",
            "class Queue:",
        ]
    },
    "algorithm": {
        "prefix": '''# Algorithm implementations

def binary_search(arr, target):
    """Binary search for target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

''',
        "prompts": [
            "def linear_search(arr, target):",
            "def bubble_sort(arr):",
            "def reverse_list(arr):",
        ]
    }
}


def detect_mode_collapse(text: str, min_repeat: int = 4) -> Dict:
    """
    Detect mode collapse patterns in generated text.

    Returns:
        Dict with 'collapsed', 'pattern', and 'score'
    """
    # Common collapse patterns
    patterns = [
        '):',    # Most common
        '::',
        '))))',
        '((((',
        '____',
        'def def',
        'self self',
        'return return',
        '########',
        '        ' * 3,  # Excessive whitespace
    ]

    for pattern in patterns:
        repeated = pattern * min_repeat
        if repeated in text:
            return {
                'collapsed': True,
                'pattern': pattern,
                'score': text.count(pattern)
            }

    # Check for any 3+ consecutive repeats of same token
    words = text.split()
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2] and len(words[i]) > 1:
            return {
                'collapsed': True,
                'pattern': f"repeated '{words[i]}'",
                'score': 1
            }

    return {'collapsed': False, 'pattern': None, 'score': 0}


def generate_with_context(
    model,
    vocab,
    context: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = 'cuda:0'
) -> str:
    """Generate text with top-k sampling."""
    tokens = vocab.encode(context)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= 256:  # Max sequence length
                break

            logits = model(x)[0, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_idx]

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return vocab.decode(x[0].tolist())


def evaluate_model(
    checkpoint_path: str,
    arch_json_path: str,
    tokenizer_path: str,
    device: str = 'cuda:0',
    samples_per_prompt: int = 3
) -> Dict:
    """
    Evaluate model on few-shot prompts.

    Returns:
        Dict with evaluation results
    """
    # Load model
    with open(arch_json_path) as f:
        arch_data = json.load(f)
    arch_cfg = ArchitectureConfig.from_dict(arch_data['architecture'])

    vocab = CodeTokenVocab(tokenizer_path=tokenizer_path)
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    model = build_model(arch_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', 'unknown')
    val_loss = ckpt.get('val_loss', 0)

    print(f"Loaded checkpoint from step {step} (val_loss={val_loss:.4f})")
    print("="*70)

    results = {
        'checkpoint': checkpoint_path,
        'step': step,
        'val_loss': val_loss,
        'categories': {},
        'summary': {
            'total_samples': 0,
            'collapsed_samples': 0,
            'collapse_rate': 0.0
        }
    }

    total_samples = 0
    collapsed_samples = 0

    for category, template in FEW_SHOT_TEMPLATES.items():
        print(f"\n[{category.upper()}]")
        print("-"*70)

        category_results = []

        for prompt in template['prompts']:
            full_context = template['prefix'] + prompt

            for sample_idx in range(samples_per_prompt):
                generated = generate_with_context(
                    model, vocab, full_context,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_k=40,
                    device=device
                )

                output_only = generated[len(full_context):]
                collapse_info = detect_mode_collapse(output_only)

                total_samples += 1
                if collapse_info['collapsed']:
                    collapsed_samples += 1

                result = {
                    'prompt': prompt,
                    'sample_idx': sample_idx,
                    'output': output_only[:150],
                    'collapsed': collapse_info['collapsed'],
                    'collapse_pattern': collapse_info['pattern']
                }
                category_results.append(result)

                # Print summary
                status = "COLLAPSE" if collapse_info['collapsed'] else "OK"
                print(f"  {prompt[:30]:30s} [{status:8s}] {repr(output_only[:50])}")

        results['categories'][category] = category_results

    # Calculate summary
    results['summary']['total_samples'] = total_samples
    results['summary']['collapsed_samples'] = collapsed_samples
    results['summary']['collapse_rate'] = collapsed_samples / total_samples if total_samples > 0 else 0

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples: {total_samples}")
    print(f"Collapsed samples: {collapsed_samples}")
    print(f"Collapse rate: {results['summary']['collapse_rate']*100:.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot evaluation for CodeLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--arch_json", type=str, required=True,
                       help="Path to architecture JSON")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--samples", type=int, default=3,
                       help="Samples per prompt")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        arch_json_path=args.arch_json,
        tokenizer_path=args.tokenizer,
        device=args.device,
        samples_per_prompt=args.samples
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
