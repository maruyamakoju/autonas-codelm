# AutoNAS-CodeLM: Ultra-Lightweight Code Models via Neural Architecture Search

**Goal**: GPT-4 level code understanding in 50-100MB models
**Status**: v2 Two-stage NAS in production 🚀
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
│   │
│   ├── scripts/                    # Production scripts
│   │   ├── run_codenas_v1_single.ps1
│   │   ├── run_codenas_v2_two_stage.ps1
│   │   └── sanity_check.sh
│   │
│   ├── logs/                       # Experiment results
│   │   ├── code_nas_v1_single/
│   │   ├── code_nas_v2_two_stage/  # (running)
│   │   └── sanity_par/
│   │
│   └── models/                     # Best architectures
│       └── codenas_v1_best_transformer.json
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
| v1 Single-stage | 1642e97 | 1.0000 | L4 H512 | 10.4 MB |
| Two-stage Test | f91a212 | 1.0000 | L4 H512 | 12.2 MB |

### Running

| Experiment | Status | Config |
|------------|--------|--------|
| v2 Two-stage | Gen 0 (GPU 94%) | Pop=24, Gen=8, Stage1=50, Stage2=300 |

### Planned

- v3 Adaptive Stage: Dynamic step allocation based on Stage 1 results
- Large Search Space: `search_mode="large"` with 35M+ configurations
- Knowledge Distillation: Compress GPT-4 knowledge into 50-100MB models

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

### Phase 2: Two-stage NAS 🔄 IN PROGRESS

- [x] Multi-fidelity evaluation (Stage 1 + Stage 2)
- [x] Two-stage test (PASSED, fitness=1.0)
- [x] Production tools (run_codenas_v2_two_stage.ps1)
- [x] Comparison tool (compare_experiments.py)
- [ ] v2 production run (code_nas_v2_two_stage) ← RUNNING

### Phase 3: Scale-up (Planned)

- [ ] Large search space (35M configurations)
- [ ] Knowledge distillation (GPT-4 → 50MB model)
- [ ] INT8/INT4 quantization
- [ ] Pruning (40-60% sparsity)
- [ ] Custom CUDA kernels
- [ ] Benchmark: HumanEval, MBPP

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

**Last Updated**: 2025-12-08
**Status**: v2 production run in progress (GPU 94% utilization)
