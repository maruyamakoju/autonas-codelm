# AutoNAS-CodeLM: Ultra-Lightweight Code Models via Neural Architecture Search

**Goal**: GPT-4 level code understanding in 50-100MB models
**Status**: Phase 3 complete - v1 model ready (few-shot collapse 14.5%) ✅
**Hardware**: RTX 5090 + RTX 4090

---

## 🎯 Project Overview

This project implements **multi-objective Neural Architecture Search (NAS)** to find optimal transformer architectures for code understanding. Unlike traditional NAS (Google ENAS, Facebook DARTS) which optimize only accuracy, we explicitly balance:

- **Accuracy** (50% weight)
- **Model Size** (30% weight, target 50-100MB)
- **Inference Latency** (20% weight, target <10ms)

---

## 🚀 Quick Start

### v2 Two-stage NAS (Recommended)

**Multi-fidelity NAS**: Fast screening (Stage 1) → Detailed evaluation (Stage 2)

```powershell
cd nas\scripts
.\run_codenas_v2_two_stage.ps1
```

**Benefits**:
- 58% faster than v1 (3000 vs 7200 training steps)
- Stage 1 (50 steps): Screen 24 architectures → ~1 min/arch
- Stage 2 (300 steps): Evaluate top 6 → ~3-5 min/arch

### Try the Current Best Model (Playground)

```bash
cd nas
python eval_playground.py
```

**Usage**: Type a code snippet and the model will generate the continuation. Interactive REPL for testing code completion.

### Quick Status (All Experiments)

```bash
cd nas
python quick_status.py
```

**Output**: Status, fitness, architecture, and metrics for all experiments.

### Compare v1 vs v2

```bash
cd nas
python compare_experiments.py
```

### BigData Training (Fix Mode Collapse)

**Issue**: Current model suffers from mode collapse (85.2%) due to small dataset (1.3MB)

**Solution**: Scale up training data to 100MB-1GB

```bash
# Step 1: Collect large-scale Python corpus
# See: data/DATA_COLLECTION_GUIDE.md

# Option A: Clone popular repos (quick start, ~50-100MB)
cd data/raw_python
git clone https://github.com/psf/requests.git
git clone https://github.com/pallets/flask.git
git clone https://github.com/django/django.git
# ... (see DATA_COLLECTION_GUIDE.md for full list)

# Step 2: Convert to char-level corpus
cd ../../nas
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --train_out ../data/code_char_big/train.txt \
  --val_out ../data/code_char_big/val.txt \
  --val_ratio 0.01

# Step 3: Train with big data (100K steps, ~1-2 hours)
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_bigdata_char \
  --train_path ../data/code_char_big/train.txt \
  --val_path ../data/code_char_big/val.txt \
  --max_steps 100000 \
  --log_dir logs/train_v1_bigdata_char \
  --device cuda:0

# Step 4: Re-evaluate
python eval_playground.py \
  --checkpoint logs/train_v1_bigdata_char/v1_bigdata_char_best.pt \
  --eval_file eval/prompts/simple_python.txt \
  --output eval/results_bigdata.jsonl
```

**Expected improvement**: Mode collapse 85.2% → <30%, Python keywords 0% → >50%

---

## 📊 Current Results

### v1 Single-stage NAS (✅ Completed)

**Best Architecture**: 4-layer Transformer
**Config**: L4 H256 Heads=8 FFN×3.0 gelu rope
**Metrics**:
- Fitness: 1.0000 🏆
- Parameters: 2.68M
- Model Size: 3.06 MB ⭐ Most lightweight!
- Val Loss: 0.0188
- Val Perplexity: 1.02
- Accuracy: 98.14%
- Latency: 3.01 ms (RTX 5090)
- Train Time: 5.03s (300 steps)

**Commit**: `1642e97` - "CodeNAS v1 complete: best 2.68M param transformer (fitness=1.0)"

**Current Production Model**: `nas/models/codenas_best_current.json` (v1 single-stage)

**Trained Model**: `logs/train_v1_production/v1_production_best.pt`
- Val Loss: 0.0065, Val PPL: 1.01
- Training: 10,000 steps, 9.6 min
- Ready for inference with `eval_playground.py`

### v2 Two-stage NAS (✅ Completed)

**Best Architecture**: 4-layer Transformer
**Config**: L4 H384 Heads=4 FFN×2.0 silu rope
**Metrics**:
- Fitness: 1.0000 🏆
- Parameters: 4.80M
- Model Size: 7.32 MB
- Val Loss: 0.0142 (better than v1!)
- Val Perplexity: 1.01
- Accuracy: 98.59% (better than v1!)
- Latency: 3.53 ms
- Train Time: 3.52s

**Conclusion**: v2 achieves higher accuracy but v1 is more lightweight (3.06MB vs 7.32MB)

### Evaluation Phase (✅ Completed)

**Batch Evaluation Results** (54 Python prompts):
- Mode collapse rate: **85.2%** ⚠️
- Avg repetition ratio: **92.22%**
- Completions with valid Python: **0%**

**Root Cause**: Character-level modeling with insufficient training data
- Current dataset: 1.3MB Python code
- Required: 100MB+ for char-level models
- Model generates repetitive patterns (`:::::`, `,,,,,`) instead of valid code

**Status**:
- ✅ NAS infrastructure complete (v1 + v2)
- ✅ Production training complete (10K steps)
- ✅ Evaluation tools built (eval_playground.py, inspect_results.py)
- ⚠️ Generation quality needs improvement

**See**: [nas/eval/EVALUATION_SUMMARY.md](nas/eval/EVALUATION_SUMMARY.md) for detailed analysis

---

## 🛠️ Key Features

### 1. Two-stage Evaluation (Multi-fidelity)

**Stage 1**: Fast screening with few training steps
**Stage 2**: Detailed evaluation for top-k candidates

```python
# Stage 1: Screen 24 architectures (50 steps each)
top_6 = screen_architectures(population, steps=50)

# Stage 2: Evaluate top 6 (300 steps each)
best = evaluate_detailed(top_6, steps=300)
```

### 2. Parallel Evaluation

- RTX 5090 + RTX 4090 dual GPU support
- Concurrent architecture evaluation
- GPU calibration for balanced workload

### 3. Production-ready Tools

```powershell
# v1 Single-stage
.\run_codenas_v1_single.ps1

# v2 Two-stage (recommended)
.\run_codenas_v2_two_stage.ps1 -Population 32 -Generations 10 -TopK 8
```

### 4. Comprehensive Comparison

```bash
python compare_experiments.py logs/code_nas_v1_single logs/code_nas_v2_two_stage
```

**Output**:
```
======================================================================
NAS Experiment Comparison
======================================================================

Metric               code_nas_v1_single   code_nas_v2_two_stage
----------------------------------------------------------------------
Fitness                          1.0000                   1.0000
Val Loss                         0.6931                   0.6931
Val PPL                           2.00                     2.00
Accuracy                       100.00%                  100.00%
Params                          2,684                    2,684
Model Size (MB)                 10.40                    10.40
Latency (ms)                     1.52                     1.52
```

---

## 📁 Repository Structure

```
1205muzi5090/
├── README.md                       # This file (project overview)
├── PROJECT_MASTER_PLAN.md          # Long-term roadmap
├── QUICKSTART.md                   # Setup instructions
│
├── nas/                            # NAS implementation
│   ├── README.md                   # Detailed NAS documentation
│   ├── EXPERIMENTS.md              # Experiment command reference
│   ├── NAS_DESIGN.md               # Technical design
│   ├── PROGRESS.md                 # Progress tracking
│   │
│   ├── evolution.py                # Evolutionary NAS (2-stage support)
│   ├── evaluator.py                # Architecture evaluation
│   ├── search_space.py             # Search space definition
│   ├── models.py                   # Model implementations
│   ├── compare_experiments.py      # v1 vs v2 comparison tool
│   ├── quick_status.py             # Quick experiment status summary
│   ├── eval_playground.py          # Interactive code completion playground
│   ├── visualize_architecture.py   # Architecture visualization tool
│   │
│   ├── scripts/                    # Production scripts
│   │   ├── run_codenas_v1_single.ps1
│   │   ├── run_codenas_v2_two_stage.ps1
│   │   └── sanity_check.sh
│   │
│   ├── eval/                       # Evaluation pipeline
│   │   ├── EVALUATION_SUMMARY.md   # Evaluation phase analysis
│   │   ├── inspect_results.py      # Batch evaluation analyzer
│   │   ├── prompts/                # Benchmark prompts
│   │   │   └── simple_python.txt   # 54 Python patterns
│   │   └── results.jsonl           # Batch evaluation results
│   │
│   ├── logs/                       # Experiment results
│   │   ├── code_nas_v1_single/     # ✅ Complete
│   │   ├── code_nas_v2_two_stage/  # ✅ Complete
│   │   ├── train_v1_production/    # ✅ Complete (10K steps)
│   │   └── sanity_par/
│   │
│   └── models/                     # Best architectures
│       └── codenas_best_current.json  # v1 production model
│
├── data/                           # Training data
│   └── code_char/
│       ├── train.txt
│       └── val.txt
│
└── .venv/                          # Python environment
```

---

## 🧪 Experiments

### Completed

| Experiment | Commit | Best Fitness | Architecture | Size |
|------------|--------|--------------|--------------|------|
| v1 Single-stage | 1642e97 | 1.0000 | L4 H256 | 3.06 MB |
| v2 Two-stage | Current | 1.0000 | L4 H384 | 7.32 MB |
| Evaluation Phase | Current | N/A | Batch eval + analysis | - |

### Next Steps

**Immediate** (to fix mode collapse):
- Option 1: Scale up training data (1.3MB → 100MB+ Python code)
- Option 2: Switch to token-level modeling (BPE/SentencePiece)
- Option 3: Knowledge distillation from GPT-4

**Future**:
- v3 Adaptive Stage: Dynamic step allocation based on Stage 1 results
- Large Search Space: `search_mode="large"` with 35M+ configurations
- Advanced techniques: Pruning, quantization, custom CUDA kernels

---

## 📈 Performance Comparison

### v1 vs v2 (Expected)

| Metric | v1 Single-stage | v2 Two-stage | Improvement |
|--------|-----------------|--------------|-------------|
| **Total Steps** | 7200 (24×300) | 3000 (24×50+6×300) | 58% reduction ⬇️ |
| **Time/Generation** | ~15 min | ~6-8 min | 2× faster ⚡ |
| **Top-k Precision** | N/A | Top 6/24 | Better focus 🎯 |
| **Best Fitness** | 1.0000 | TBD | TBD |

---

## 🔧 Requirements

```bash
# Core dependencies
torch==2.8.0
transformers==4.53.0
numpy
matplotlib

# GPU
CUDA 12.x
cuDNN 9.x
```

**Hardware**:
- GPU: RTX 5090 (24GB) or RTX 4090 (24GB)
- RAM: 32GB+ recommended
- Storage: 10GB+ for logs and checkpoints

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview (this file) |
| [nas/README.md](nas/README.md) | Detailed NAS documentation |
| [nas/EXPERIMENTS.md](nas/EXPERIMENTS.md) | Experiment command reference |
| [nas/NAS_DESIGN.md](nas/NAS_DESIGN.md) | Technical design document |
| [nas/eval/EVALUATION_SUMMARY.md](nas/eval/EVALUATION_SUMMARY.md) | Evaluation phase analysis |
| [PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md) | Long-term roadmap |
| [QUICKSTART.md](QUICKSTART.md) | Setup instructions |

---

## 🚦 Project Status

### Phase 1: Core NAS System ✅ COMPLETED

- [x] Search space design (35M+ configurations)
- [x] Evolutionary NAS with genetic algorithm
- [x] Multi-objective fitness (accuracy, size, latency)
- [x] Transformer model implementation
- [x] Parallel evaluation (RTX 5090 + 4090)
- [x] Production scripts (PowerShell + Bash)

### Phase 2: Two-stage NAS ✅ COMPLETED

- [x] Multi-fidelity evaluation (Stage 1 + Stage 2)
- [x] Two-stage test (PASSED, fitness=1.0)
- [x] Production tools (run_codenas_v2_two_stage.ps1)
- [x] Comparison tool (compare_experiments.py)
- [x] v2 production run (fitness=1.0, L4 H384, 7.32MB)
- [x] v1 production training (10K steps, val_loss=0.0065)
- [x] Evaluation pipeline (batch eval, inspection tools)

**Result**: Both v1 and v2 achieve fitness=1.0, but generation quality suffers from mode collapse due to insufficient training data (1.3MB Python code → need 100MB+)

### Phase 3: Quality Improvement ✅ COMPLETED

**Goal**: Fix mode collapse through strong regularization + token-level modeling

**Achievement**: Reduced collapse rate from **85%+** to **14.5%** with few-shot prompts

**Solution Applied**:
- [x] **8K BPE Tokenization** (3.4x compression vs char-level)
- [x] **Strong Regularization** (label_smoothing=0.1, dropout=0.2, weight_decay=0.05)
- [x] **Larger Model** (29M params, L8 H512)
- [x] **Extended Training** (50K steps, 100MB Python corpus)
- [x] **Few-shot Evaluation Framework** (eval/few_shot_eval.py)

**Current Best Model (v1)**:
- **Config**: `nas/models/codenas_l8h512_regularized.json`
- **Checkpoint**: `nas/logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt`
- **Tokenizer**: `data/tokenizers/python_bpe_8k/tokenizer.json`
- **Size**: 55.6 MB (29M params)
- **Val Loss**: 1.254, PPL: 3.50
- **Collapse Rate**: 14.5% (few-shot), ~90% (short prompts)

**Key Finding**: Mode collapse is **context-dependent**. Model requires 1-2 example functions to avoid repetition patterns.

**Usage**: See [STRONGREG_SUMMARY.md](STRONGREG_SUMMARY.md) for detailed results and usage guidelines

### Phase 4: Advanced Optimization (Planned)

- [ ] INT8/INT4 quantization
- [ ] Structured pruning (40-60% sparsity)
- [ ] Custom CUDA kernels for inference
- [ ] Production deployment (ONNX, TensorRT)
- [ ] Comprehensive benchmarks (HumanEval, MBPP, CodeXGLUE)

---

## 🎓 Research Context

### What's Different from Literature?

| Feature | This Work | Google ENAS | Facebook DARTS |
|---------|-----------|--------------|----------------|
| **Objectives** | 3 (acc+size+lat) | 1 (accuracy) | 1 (accuracy) |
| **Search Method** | Genetic Algorithm | Reinforcement Learning | Gradient-based |
| **Evaluation** | Multi-fidelity (2-stage) | Single-fidelity | Single-fidelity |
| **Target** | 50-100MB models | Accuracy maximization | Accuracy maximization |
| **Hardware** | Local GPU (RTX 5090) | TPU cluster | GPU cluster |
| **Cost** | ~$0 (local) | $10,000s | $10,000s |

**Our Advantages**:
1. ✅ Explicit size & latency optimization
2. ✅ Multi-fidelity evaluation (58% speedup)
3. ✅ More interpretable (genetic vs RL/gradients)
4. ✅ Zero marginal cost (local GPU)

---

## 📝 Recent Commits

```
8216980 Add v2 production tools: PowerShell launcher + docs
f91a212 Add Two-stage NAS (Multi-fidelity) support
1642e97 CodeNAS v1 complete: best 2.68M param transformer (fitness=1.0)
f5829be NAS infra complete: parallel eval + sanity check (seq vs par)
```

---

## 📧 Contact

**Author**: Koju
**Project**: AutoNAS-CodeLM
**Date**: December 2025
**GPU**: RTX 5090 + RTX 4090

---

## 📜 License

MIT License (for research and commercial use)

---

## 🙏 Acknowledgments

- **Papers**: Transformer (Vaswani+ 2017), ENAS (Pham+ 2018), DARTS (Liu+ 2019), LLaMA (Touvron+ 2023)
- **Tools**: PyTorch, HuggingFace Transformers, Claude Code
- **Hardware**: NVIDIA RTX 5090 + RTX 4090

---

## 🔗 Quick Links

- [Run v2 Two-stage NAS](nas/scripts/run_codenas_v2_two_stage.ps1)
- [Compare v1 vs v2](nas/compare_experiments.py)
- [View Best Architectures](nas/models/)
- [Experiment Logs](nas/logs/)
- [Detailed Documentation](nas/README.md)

---

**Last Updated**: 2025-12-10
**Status**: Phase 3 complete - v1 model (29M params, 55.6MB) ready for production with few-shot prompts
**Achievement**: Mode collapse 85%+ → 14.5% (few-shot context)
**Usage**: See [STRONGREG_SUMMARY.md](STRONGREG_SUMMARY.md) for complete results and usage guidelines

**Quick Start (v1 Model)**:
```bash
cd nas
python eval_playground.py \
  --mode token \
  --checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
  --tokenizer_path ../data/tokenizers/python_bpe_8k/tokenizer.json \
  --arch_json models/codenas_l8h512_regularized.json
```

**⚠️ Important**: Use few-shot prompts (1-2 example functions) to avoid mode collapse. Short prompts still collapse ~90%.

**See Also**:
- [STRONGREG_SUMMARY.md](STRONGREG_SUMMARY.md) - Phase 3 complete results
- [TOKEN_LEVEL_SUMMARY.md](TOKEN_LEVEL_SUMMARY.md) - Token-level modeling analysis
- [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) - Char-level Phase 1 analysis
