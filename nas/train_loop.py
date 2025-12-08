"""
Training loop for NAS evaluation

Real training of architecture candidates on code LM task
"""

from typing import Dict
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

from search_space import ArchitectureConfig
from models import build_model
from datasets import CodeCharDatasetConfig, build_code_char_loaders


def train_one_architecture(
    arch_cfg: ArchitectureConfig,
    data_cfg: CodeCharDatasetConfig,
    device: str = "cuda:0",
    max_steps: int = 500,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    log_interval: int = 100
) -> Dict[str, float]:
    """
    Train a single architecture and return metrics

    Args:
        arch_cfg: Architecture configuration
        data_cfg: Dataset configuration
        device: Device to train on
        max_steps: Maximum training steps
        lr: Learning rate
        grad_clip: Gradient clipping value
        log_interval: Log every N steps

    Returns:
        Dictionary with metrics:
        - val_loss: Validation loss
        - val_ppl: Validation perplexity
        - train_loss: Final training loss
        - num_params: Number of parameters
        - model_size_mb: Model size in MB
        - latency_ms: Inference latency
        - train_time_s: Training time in seconds
    """
    print(f"\n[TRAIN] Starting training...")
    print(f"  Architecture: {arch_cfg.arch_type}")
    print(f"  Layers: {arch_cfg.num_layers}, Hidden: {arch_cfg.hidden_dim}")
    print(f"  Device: {device}")
    print(f"  Max steps: {max_steps}")

    # Build data loaders
    train_loader, val_loader, vocab = build_code_char_loaders(data_cfg)

    # Update architecture config with actual vocab size
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = data_cfg.seq_len

    # Build model
    print(f"\n[MODEL] Building model...")
    try:
        model = build_model(arch_cfg).to(device)
    except NotImplementedError as e:
        print(f"  [WARNING] {e}")
        print(f"  [WARNING] Using simple fallback model")
        # Use simple model
        from evaluator import Evaluator
        evaluator = Evaluator()
        model = evaluator._build_simple_model(arch_cfg)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    t0 = time.time()

    print(f"\n[TRAIN] Training for {max_steps} steps...")

    for epoch in range(100):  # Max epochs (will break early)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)  # (B, T)

            # Forward
            logits = model(x)  # (B, T, V)

            # Loss
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            # Log
            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                print(f"  Step {step}/{max_steps}: loss={avg_loss:.4f}, ppl={ppl:.2f}")
                total_loss = 0.0

            # Stop if max steps reached
            if step >= max_steps:
                break

        if step >= max_steps:
            break

    train_time = time.time() - t0
    print(f"[TRAIN] Training complete: {train_time:.1f}s")

    # Validation
    print(f"\n[EVAL] Running validation...")
    model.eval()
    val_loss = 0.0
    val_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            val_loss += loss.item() * y.numel()
            val_tokens += y.numel()

    val_loss /= val_tokens
    val_ppl = torch.exp(torch.tensor(val_loss)).item()

    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Validation perplexity: {val_ppl:.2f}")

    # Model size
    model_size_mb = arch_cfg.estimate_size_mb()

    # Latency measurement
    print(f"\n[LATENCY] Measuring inference latency...")
    model.eval()

    # Get a sample batch
    x_sample, _ = next(iter(val_loader))
    x_sample = x_sample.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_sample)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t1 = time.time()

    with torch.no_grad():
        for _ in range(100):
            _ = model(x_sample)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    latency_ms = (time.time() - t1) / 100 * 1000

    print(f"  Latency: {latency_ms:.2f}ms per batch")

    # Return metrics
    metrics = {
        "val_loss": float(val_loss),
        "val_ppl": float(val_ppl),
        "train_loss": float(total_loss / max(log_interval, 1)),
        "num_params": float(num_params),
        "model_size_mb": float(model_size_mb),
        "latency_ms": float(latency_ms),
        "train_time_s": float(train_time),
    }

    print(f"\n[METRICS] Summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    from search_space import get_baseline_architectures

    print("="*60)
    print("Training Loop Test")
    print("="*60)

    # Get a small baseline
    baselines = get_baseline_architectures()
    arch_cfg = baselines[2]  # Ultra-small model

    # Data config
    data_cfg = CodeCharDatasetConfig(
        train_path="../data/code_char/train.txt",
        val_path="../data/code_char/val.txt",
        seq_len=128,
        batch_size=16
    )

    # Train
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metrics = train_one_architecture(
        arch_cfg,
        data_cfg,
        device=device,
        max_steps=100,  # Quick test
        lr=3e-4
    )

    print("\n" + "="*60)
    print("Training test complete!")
    print("="*60)
