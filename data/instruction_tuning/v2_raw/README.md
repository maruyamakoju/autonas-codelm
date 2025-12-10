# v2 Raw Instruction Data

Converted datasets for v2 instruction tuning, before any filtering or merging.

## Files

- `humaneval_supervised.jsonl` - All 164 HumanEval tasks
- `mbpp_supervised.jsonl` - Filtered MBPP tasks (target: 300-500 simple tasks)
- `humaneval_sample.jsonl` - Hand-converted HumanEval samples for testing (3-5 tasks)

## Format

Each line is a JSON object:
```json
{
  "id": "task_identifier",
  "prompt": "def function(args):\n    \"\"\"Docstring.\"\"\"\n",
  "solution": "    return result\n"
}
```

**Important**:
- `prompt`: Function signature + docstring (no implementation)
- `solution`: Implementation ONLY (indented, no signature)
- Both end with `\n` for proper formatting

## Next Steps

1. Generate `humaneval_supervised.jsonl` (164 tasks)
2. Generate `mbpp_supervised.jsonl` (300-500 filtered tasks)
3. Merge into `data/instruction_tuning/v2_train.jsonl` (450 train)
4. Create `data/instruction_tuning/v2_val.jsonl` (50 val)

Target: 500 total samples for v2 training.
