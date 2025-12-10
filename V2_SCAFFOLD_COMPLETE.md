# v2 Scaffold Complete

**Date**: 2025-12-10
**Branch**: `v2-lora-500samples`
**Status**: ✅ Ready for data collection and LoRA implementation

---

## What Was Set Up

### 1. Directory Structure ✅

```
data/
├── external/              # Raw datasets from external sources
│   ├── README.md         # Download instructions for HumanEval, MBPP
│   ├── HumanEval.jsonl   # (to be downloaded)
│   └── mbpp.jsonl        # (to be downloaded)
└── instruction_tuning/
    └── v2_raw/           # Converted supervised format
        ├── README.md     # Format specification
        ├── humaneval_sample.jsonl  # 5 hand-converted samples ✅
        ├── humaneval_supervised.jsonl  # (to be generated, 164 tasks)
        └── mbpp_supervised.jsonl      # (to be generated, 300-500 tasks)

nas/
└── tools/                # Conversion scripts
    ├── convert_humaneval_to_supervised.py  # HumanEval converter ✅
    └── convert_mbpp_to_supervised.py       # MBPP converter ✅
```

### 2. Conversion Scripts ✅

**`convert_humaneval_to_supervised.py`**:
- Converts HumanEval format → our supervised JSONL
- Extracts prompt (signature + docstring) and solution (implementation)
- Maps `HumanEval/0` → `he_0` for cleaner IDs
- Ready to run once HumanEval.jsonl is downloaded

**`convert_mbpp_to_supervised.py`**:
- Converts MBPP format → our supervised JSONL
- Filters: <=30 lines, simple difficulty only
- TODO: Needs proper prompt/solution splitting logic
- Placeholder implementation ready for refinement

### 3. Sample Data ✅

**`humaneval_sample.jsonl`** - 5 hand-converted HumanEval tasks:
- `he_0`: has_close_elements (list operations)
- `he_1`: separate_paren_groups (string parsing)
- `he_2`: truncate_number (simple math)
- `he_17`: parse_music (string parsing + mapping)
- `he_18`: how_many_times (substring search)

**Format verified**:
```json
{
  "id": "he_0",
  "prompt": "from typing import List\n\n\ndef has_close_elements(...) -> bool:\n    \"\"\"...\"\"\"\n",
  "solution": "    for idx, elem in enumerate(numbers):\n        ...\n    return False\n"
}
```

---

## Next Steps

### Phase 1: Data Collection (1-2 days)

1. **Download HumanEval** (10 minutes):
   ```bash
   cd data/external
   wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
   gunzip HumanEval.jsonl.gz
   ```

2. **Download MBPP** (10 minutes):
   ```bash
   cd data/external
   wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl
   ```

3. **Convert HumanEval** (5 minutes):
   ```bash
   python nas/tools/convert_humaneval_to_supervised.py
   # Output: data/instruction_tuning/v2_raw/humaneval_supervised.jsonl (164 tasks)
   ```

4. **Fix and convert MBPP** (2-4 hours):
   - Implement proper prompt/solution splitting in `convert_mbpp_to_supervised.py`
   - Run conversion
   - Filter for 300-500 simple tasks
   - Output: `data/instruction_tuning/v2_raw/mbpp_supervised.jsonl`

5. **Merge and split** (30 minutes):
   ```bash
   # Combine HumanEval (164) + MBPP (300-500) = 464-664 total
   # Split: 450 train, 50 validation
   # Output:
   #   data/instruction_tuning/v2_train.jsonl
   #   data/instruction_tuning/v2_val.jsonl
   ```

### Phase 2: LoRA Implementation (1 day)

1. **Install PEFT library**:
   ```bash
   pip install peft
   ```

2. **Create LoRA training script**:
   - Copy `train_instruction_tuning.py` → `train_v2_lora.py`
   - Add LoRA adapters to model (rank=8, alpha=16)
   - Target modules: `qkv_proj`, `out_proj` (attention layers)
   - Freeze base model, only train LoRA weights

3. **Configure training**:
   - Init: `logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt`
   - Data: `data/instruction_tuning/v2_train.jsonl` (450-550 samples)
   - Steps: 20k (longer than v1+IT due to more data)
   - LR: 1e-4 (can be higher since base is frozen)
   - Batch size: 8

### Phase 3: Training & Evaluation (1-2 days)

1. **Run training**:
   ```bash
   python nas/train_v2_lora.py \
     --init_checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
     --arch_json models/codenas_l8h512_regularized.json \
     --tokenizer_path ../data/tokenizers/python_bpe_8k/tokenizer.json \
     --data_path ../data/instruction_tuning/v2_train.jsonl \
     --val_path ../data/instruction_tuning/v2_val.jsonl \
     --output_dir logs/train_v2_lora \
     --max_steps 20000 \
     --lr 1e-4 \
     --batch_size 8 \
     --eval_interval 2000
   ```

2. **Evaluate every 2k steps**:
   ```bash
   python eval/run_mini_benchmark.py \
     --checkpoint logs/train_v2_lora/v2_lora_step{2000,4000,...}.pt \
     --samples 1 \
     --temperature 0.8
   ```

3. **Target**: 0/20 → 5-10/20 pass rate

---

## Timeline Estimate

| Phase | Task | Time |
|-------|------|------|
| **Phase 1** | Download datasets | 20 min |
| | Convert HumanEval | 5 min |
| | Fix MBPP converter | 2-4 hours |
| | Merge & split | 30 min |
| **Phase 2** | Install PEFT | 5 min |
| | Implement LoRA wrapper | 4-6 hours |
| | Test on small data | 1 hour |
| **Phase 3** | Training (20k steps) | 12-16 hours |
| | Evaluation | 2 hours |
| **Total** | | **2-3 days** |

---

## Decision Point

After Phase 1 (data collection) is complete, re-evaluate:

**If 500+ samples collected successfully**:
→ Proceed to Phase 2 (LoRA implementation)

**If fewer than 400 samples**:
→ Consider Option C (GPT-4 synthetic generation) to supplement

**If data quality is poor**:
→ Manual curation or filtering needed

---

**Current Status**: 🚀 **Scaffold complete. Ready to start Phase 1 (data collection).**

**Next Action**: Download HumanEval and MBPP datasets.
