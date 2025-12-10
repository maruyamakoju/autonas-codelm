"""
Collate function for instruction tuning with dynamic padding.

Pads sequences to the max length in each batch (not to a fixed seq_len).
This prevents training on unnecessary padding tokens.
"""

import torch


def collate_instruction_batch(batch, pad_value=0, ignore_index=-100):
    """
    Collate function for variable-length instruction sequences.

    Args:
        batch: List of (x, y) tuples from InstructionCodeDataset
        pad_value: Value to use for padding input sequences (default: 0)
        ignore_index: Value to use for padding target sequences (default: -100)

    Returns:
        x_batch: Padded input tensor [B, max_len]
        y_batch: Padded target tensor [B, max_len]
    """
    # Unpack batch
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    # Find max length in this batch
    max_len = max(x.size(0) for x in x_list)

    batch_size = len(x_list)

    # Create padded tensors
    x_batch = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    y_batch = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)

    # Fill with actual data
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        seq_len = x.size(0)
        x_batch[i, :seq_len] = x
        y_batch[i, :seq_len] = y

    return x_batch, y_batch
