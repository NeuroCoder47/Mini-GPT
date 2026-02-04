<div align="center">

# 🚀 GPT from Scratch: Complete LLM Pretraining Pipeline

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
<img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">

### *Building Language Intelligence from the Ground Up*

[Features](#-features) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Pipeline](#-pipeline) •
[Training](#-training) •
[Results](#-results)

---

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=2E9EF7&center=true&vCenter=true&width=600&lines=Pre-training+a+GPT+Model;From+Data+to+Deployment;Tokenization+%E2%86%92+Training+%E2%86%92+Generation" alt="Typing SVG" />

</div>

---

## 📖 Overview

This project implements a **complete end-to-end pipeline** for pre-training a GPT-style language model from scratch. Unlike many educational implementations that skip crucial steps, this repository includes everything needed for production-grade LLM training:

- ✅ Custom data preprocessing pipeline
- ✅ BPE tokenizer training from scratch
- ✅ Efficient binary data storage
- ✅ Full GPT architecture implementation
- ✅ Optimized training loop with validation

<div align="center">

```mermaid
graph LR
    A[📚 Raw Text Data] -->|Preprocess| B[🔄 Shuffled Dataset]
    B -->|Tokenize| C[🎯 BPE Tokenizer]
    C -->|Encode| D[💾 Binary Files]
    D -->|Train| E[🤖 GPT Model]
    E -->|Generate| F[✨ Text Output]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#f5e1ff
    style F fill:#ffe1e1
```

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Data Processing
- **Efficient Shuffling**: Explicit index-based shuffling for reproducibility
- **Binary Storage**: Memory-mapped `.bin` files for fast training
- **Train/Val Split**: Automatic 99/1 split with separate files
- **Batch Processing**: Chunked processing to handle large datasets

</td>
<td width="50%">

### 🔤 Tokenization
- **BPE Algorithm**: Byte Pair Encoding from scratch
- **Custom Vocabulary**: Configurable vocab size (default: 32,768)
- **Special Tokens**: `<|bos|>`, `<|eos|>`, `<|pad|>`
- **Unicode Support**: Full UTF-8 compatibility
- **Source**: Tokenizer code adapted from [Nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy

</td>
</tr>
<tr>
<td width="50%">

### 🧠 Model Architecture
- **Transformer Decoder**: Pure decoder-only architecture
- **Multi-Head Attention**: Configurable attention heads
- **Position Encoding**: Sinusoidal positional embeddings
- **Layer Normalization**: Pre-norm architecture
- **Dropout Regularization**: Prevent overfitting

</td>
<td width="50%">

# 🚀 GPT Training Pipeline

> Production-grade autoregressive language model training with efficient mixed precision and streaming data

---

## ⚙️ Model Config

```yaml
Architecture: 4-layer GPT
Heads: 16 | Dims: 512 | Params: ~25M
Vocab: 32K tokens | Context: 512 tokens
Training: 10K steps | Batch: 8→32 (grad accumulation)
```

---

## ✨ Core Features

### 🎯 **Mixed Precision Training**
Trains in FP16 for 2x speed while maintaining FP32 master weights. Automatic gradient scaling prevents tiny gradient underflow and dynamically adapts to avoid overflow.

### 📦 **Gradient Accumulation** 
Simulates large batches (32) on limited VRAM by accumulating gradients over 4 micro-batches before weight updates.

### 📈 **Cosine LR Schedule**
Linear warmup (50 steps) followed by smooth cosine decay from 6e-4 to 6e-5 over full training run.

### ⚡ **Optimized Data Loading**
- **DataTrove .ds format**: Binary memory-mapped tokenized sequences for instant access
- **Cyclic iterators**: Infinite streaming without epoch boundaries using `cycle()`
- **Prefetching pipeline**: 2 workers + 2 prefetch factor for compute/IO overlap
- **Persistent workers**: Processes stay alive to eliminate reload overhead
- **Pinned memory**: Non-blocking GPU transfers during computation

### 🔍 **Training Monitoring**
- **Real-time tracking**: Loss, learning rate, MFU% logged every 10 steps
- **Validation probes**: Eval runs every 100 steps to monitor generalization
- **Hardware metrics**: Model FLOPS Utilization tracks GPU efficiency

### 🎲 **Autoregressive Training**
Next-token prediction using teacher forcing. Input shifted by one position to learn P(token_t | context).

### 🛡️ **Stability Guards**
- Gradient clipping (max norm 1.0) prevents exploding gradients
- Weight decay (0.1) for regularization
- Dynamic loss scaling adjusts to prevent FP16 overflow/underflow

---

## 🎓 Training Flow

```
Dynamic LR → Grad Accumulation (4 micro-steps) → Mixed Precision
    ↓              ↓                                    ↓
Per-step    Simulate batch=32         FP16 compute, FP32 params
adjustment   with batch=8 memory      + auto gradient scaling
    ↓              ↓                                    ↓
Gradient Clipping → Weight Update → Validation Probes (every 100)
```

---

## 💡 Implementation Highlights

**Performance**
- Fused AdamW optimizer for kernel-level speedups
- Ready for `torch.compile()` (2x boost when uncommented)
- Non-blocking CUDA transfers overlap data loading with compute

**Data Pipeline**  
- `.ds` files use 2-byte token encoding supporting 65K vocab
- StopIteration-free loading via cyclic sampling
- Automatic handling of variable dataset sizes

**Mixed Precision Details**
- Autocast manages FP16/FP32 casting automatically
- Gradient scaler handles loss multiplication, backprop, unscaling, and overflow detection
- Dynamic scale factor adjustment (halves on overflow, doubles after 2K successful steps)

---

## 📊 What to Explore

**Dive into the code to understand:**
- How gradient scaler prevents FP16 underflow while avoiding overflow
- Why loss is divided by accumulation steps before backward pass
- MFU calculation methodology (FLOPS achieved vs theoretical peak)
- How cyclic iterators enable infinite training without data reloading
- The complete mixed precision flow from forward pass to parameter update

**Inline comments explain:**
- Full gradient scaling mechanics (scaling → backward → unscaling → update)
- Loss averaging strategy across micro-batches
- Dynamic scale factor adjustment logic

---

## 🎯 Quick Reference

| Component | Value | Purpose |
|-----------|-------|---------|
| **Batch Size** | 8 | GPU memory constraint |
| **Grad Accum Steps** | 4 | Effective batch = 32 |
| **Learning Rate** | 6e-4 → 6e-5 | Cosine decay |
| **Warmup** | 50 steps | Early stability |
| **Weight Decay** | 0.1 | L2 regularization |
| **Grad Clip** | 1.0 | Explosion prevention |
| **Workers** | 2 | Data loading parallelism |

---

<div align="center">

**🔬 Explore the code to see how production-grade LLM training works!**

Built for learning and experimentation 🚀

</div>
