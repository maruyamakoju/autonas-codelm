"""
Debug script to verify dataset x/y shift for next-token prediction
"""
import sys
sys.path.insert(0, '..')

from datasets import CodeTokenDatasetConfig, build_code_token_loaders

print("=" * 70)
print("Dataset Shift Verification")
print("=" * 70)

# Create dataset
cfg = CodeTokenDatasetConfig(
    train_path="../../data/code_token_bigdata/train.txt",
    val_path="../../data/code_token_bigdata/val.txt",
    seq_len=32,
    batch_size=4,
    tokenizer_path="../../data/tokenizers/python_bpe_8k/tokenizer.json"
)

train_loader, val_loader, vocab = build_code_token_loaders(cfg)

# Get first batch
x, y = next(iter(train_loader))

print(f"\nBatch shape: x={x.shape}, y={y.shape}")
print(f"\nFirst sequence (first 16 tokens):")
print(f"x[0]: {x[0, :16].tolist()}")
print(f"y[0]: {y[0, :16].tolist()}")

print(f"\nVerifying shift:")
print(f"y[0,0] should equal x[0,1]: y={y[0,0].item()}, x={x[0,1].item()} -> {'✓ MATCH' if y[0,0]==x[0,1] else '✗ MISMATCH'}")
print(f"y[0,1] should equal x[0,2]: y={y[0,1].item()}, x={x[0,2].item()} -> {'✓ MATCH' if y[0,1]==x[0,2] else '✗ MISMATCH'}")
print(f"y[0,2] should equal x[0,3]: y={y[0,2].item()}, x={x[0,3].item()} -> {'✓ MATCH' if y[0,2]==x[0,3] else '✗ MISMATCH'}")

# Decode and show
print(f"\nDecoded content:")
x_text = vocab.decode(x[0, :16].tolist())
y_text = vocab.decode(y[0, :16].tolist())
print(f"x (input):  {repr(x_text)}")
print(f"y (target): {repr(y_text)}")

print("\n" + "=" * 70)
print("If y[0,0]==x[0,1] and y[0,1]==x[0,2], dataset shift is CORRECT")
print("=" * 70)
