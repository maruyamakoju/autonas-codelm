# External Datasets

This directory contains raw datasets downloaded from external sources.

## HumanEval

**Source**: https://github.com/openai/human-eval

**Format**: JSONL with fields:
- `task_id`: Unique identifier (e.g., "HumanEval/0")
- `prompt`: Function signature + docstring
- `canonical_solution`: Reference implementation
- `test`: Test cases
- `entry_point`: Function name

**Download**:
```bash
# Option 1: Direct download
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz

# Option 2: Clone repo
git clone https://github.com/openai/human-eval.git
cp human-eval/data/HumanEval.jsonl.gz .
gunzip HumanEval.jsonl.gz
```

**Total tasks**: 164

---

## MBPP (Mostly Basic Python Problems)

**Source**: https://github.com/google-research/google-research/tree/master/mbpp

**Format**: JSONL with fields:
- `task_id`: Integer ID
- `text`: Problem description
- `code`: Reference solution
- `test_list`: List of test cases
- `test_setup_code`: Setup code for tests
- `challenge_test_list`: Additional tests

**Download**:
```bash
# Download sanitized version
wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl

# Or full version with test cases
wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp_test.jsonl
```

**Total tasks**: ~1000 (filter for "simple" difficulty)

---

## Usage

After downloading datasets here, use conversion scripts in `nas/tools/` to convert them to our supervised format:

```bash
# Convert HumanEval
python nas/tools/convert_humaneval_to_supervised.py

# Convert MBPP
python nas/tools/convert_mbpp_to_supervised.py
```

Output will be written to `data/instruction_tuning/v2_raw/`.
