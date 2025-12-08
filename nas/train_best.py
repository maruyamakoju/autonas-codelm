"""
Train the best architecture found by NAS

Full training run with:
- Long training (5,000-20,000 steps)
- Learning rate scheduling
- Checkpointing
- Detailed logging
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

from search_space import ArchitectureConfig
from models import build_model
from datasets import CodeCharDatasetConfig, build_code_char_loaders


class TrainingLogger:
    """Training metrics logger"""

    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.metrics: List[Dict] = []
        self.log_file = log_dir / f"{experiment_name}_log.jsonl"

    def log(self, step: int, metrics: Dict):
        """Log metrics at given step"""
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self.metrics.append(entry)

        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_summary(self, final_metrics: Dict):
        """Save final summary"""
        summary = {
            "experiment_name": self.experiment_name,
            "final_metrics": final_metrics,
            "total_steps": len(self.metrics),
            "all_metrics": self.metrics
        }

        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[LOG] Saved summary to {summary_file}")


def train_best_architecture(
    arch_cfg: ArchitectureConfig,
    data_cfg: CodeCharDatasetConfig,
    experiment_name: str = "best_arch",
    device: str = "cuda:0",
    max_steps: int = 10000,
    lr: float = 3e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    log_interval: int = 100,
    eval_interval: int = 500,
    save_interval: int = 2000,
    log_dir: str = "logs/train_best"
) -> Dict:
    """
    Full training of best architecture

    Features:
    - Cosine LR schedule with warmup
    - Periodic evaluation
    - Checkpoint saving
    - Detailed logging

    Returns:
        Final metrics dictionary
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(log_path, experiment_name)

    print("="*70)
    print("BEST ARCHITECTURE TRAINING")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {lr} -> {min_lr}")
    print(f"Warmup: {warmup_steps} steps")

    # Build data loaders
    train_loader, val_loader, vocab = build_code_char_loaders(data_cfg)

    # Update vocab size
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = data_cfg.seq_len

    # Build model
    print(f"\n[MODEL] Architecture: {arch_cfg.arch_type}")
    print(f"  Layers: {arch_cfg.num_layers}, Hidden: {arch_cfg.hidden_dim}")
    print(f"  Heads: {arch_cfg.num_heads}, FFN mult: {arch_cfg.ffn_multiplier}")
    print(f"  Norm: {arch_cfg.normalization}, Act: {arch_cfg.activation}")
    print(f"  Position: {arch_cfg.position_encoding}")

    model = build_model(arch_cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Estimated size: {arch_cfg.estimate_size_mb():.2f} MB")

    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # LR scheduler: cosine with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr/lr + (1 - min_lr/lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss()

    # Training state
    step = 0
    total_loss = 0.0
    best_val_loss = float('inf')
    t0 = time.time()

    print(f"\n[TRAIN] Starting training for {max_steps} steps...")
    print("-"*70)

    model.train()

    for epoch in range(1000):  # Will break on max_steps
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                grad_norm = 0.0
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1
            current_lr = optimizer.param_groups[0]['lr']

            # Logging
            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                ppl = math.exp(min(avg_loss, 10))  # Clip for stability
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed

                print(f"Step {step:6d}/{max_steps} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:8.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"{steps_per_sec:.1f} steps/s")

                logger.log(step, {
                    "train_loss": avg_loss,
                    "train_ppl": ppl,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm),
                    "steps_per_sec": steps_per_sec
                })

                total_loss = 0.0

            # Evaluation
            if step % eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)
                print(f"  [EVAL] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                logger.log(step, {
                    "val_loss": val_loss,
                    "val_ppl": val_ppl
                })

                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, val_loss,
                                   log_path / f"{experiment_name}_best.pt")
                    print(f"  [BEST] New best model saved!")

                model.train()

            # Save checkpoint
            if step % save_interval == 0:
                save_checkpoint(model, optimizer, step, val_loss,
                               log_path / f"{experiment_name}_step{step}.pt")

            # Stop condition
            if step >= max_steps:
                break

        if step >= max_steps:
            break

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device)
    train_time = time.time() - t0

    # Latency measurement
    latency_ms = measure_latency(model, val_loader, device)

    final_metrics = {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "best_val_loss": best_val_loss,
        "num_params": num_params,
        "model_size_mb": arch_cfg.estimate_size_mb(),
        "latency_ms": latency_ms,
        "total_steps": step,
        "train_time_s": train_time,
        "train_time_min": train_time / 60,
        "architecture": arch_cfg.to_dict()
    }

    print(f"\nFinal Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val PPL: {val_ppl:.2f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Model Size: {arch_cfg.estimate_size_mb():.2f} MB")
    print(f"  Latency: {latency_ms:.2f} ms")
    print(f"  Training Time: {train_time/60:.1f} min")

    # Save final checkpoint
    save_checkpoint(model, optimizer, step, val_loss,
                   log_path / f"{experiment_name}_final.pt")

    logger.save_summary(final_metrics)

    return final_metrics


def evaluate(model, val_loader, loss_fn, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 10))

    return avg_loss, ppl


def measure_latency(model, val_loader, device, num_runs=100):
    """Measure inference latency"""
    model.eval()
    x, _ = next(iter(val_loader))
    x = x.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    latency = (time.time() - t0) / num_runs * 1000
    return latency


def save_checkpoint(model, optimizer, step, val_loss, path):
    """Save training checkpoint"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss
    }, path)


def load_best_architecture(json_path: str) -> ArchitectureConfig:
    """Load architecture from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle nested structure (from evolution output)
    if "architecture" in data:
        arch_dict = data["architecture"]
    else:
        arch_dict = data

    return ArchitectureConfig.from_dict(arch_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train best architecture")
    parser.add_argument("--arch_json", type=str,
                       default="logs/code_char_sanity/evolution/best_architecture.json")
    parser.add_argument("--experiment_name", type=str, default="best_L6_H512")
    parser.add_argument("--train_path", type=str, default="../data/code_char/train.txt")
    parser.add_argument("--val_path", type=str, default="../data/code_char/val.txt")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, default="logs/train_best")
    args = parser.parse_args()

    # Load best architecture
    print(f"Loading architecture from: {args.arch_json}")
    arch_cfg = load_best_architecture(args.arch_json)

    # Data config
    data_cfg = CodeCharDatasetConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )

    # Train
    final_metrics = train_best_architecture(
        arch_cfg=arch_cfg,
        data_cfg=data_cfg,
        experiment_name=args.experiment_name,
        device=args.device,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_dir=args.log_dir
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
