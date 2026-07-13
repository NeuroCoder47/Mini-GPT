
<div align="center">

# 🚀 GPT from Scratch: Complete LLM Pretraining Pipeline

<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
<img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">

### *Building Language Intelligence from the Ground Up*

[Overview](#-overview) - 
[Architecture](#-architecture) - 
[Quick Start](#-quick-start) - 
[Training](#-training) - 
[Results](#-results)

***

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=2E9EF7&center=true&vCenter=true&width=600&lines=Pre-training+a+GPT+Model;From+Data+to+Deployment;Tokenization+%E2%86%92+Training+%E2%86%92+Generation" alt="Typing SVG" />

</div>

***

## 📖 Overview

This project implements a complete end-to-end pipeline for pre-training a GPT-style language model from scratch. Unlike educational implementations that skip crucial steps, this repository includes everything needed for production-grade LLM training: custom data preprocessing, BPE tokenizer training, efficient binary storage, full transformer architecture, and optimized training with mixed precision.

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

***

## 🏗️ Architecture

### Model Configuration

```yaml
Architecture: 12-layer GPT Transformer Decoder
Attention Heads: 12 | Embedding Dims: 768
Parameters: ~125M | Context Window: 1024 tokens
Vocabulary: 50,257
Head Dimension: 64
Feed-Forward Dimension: 3072
```

### Core Components

**Transformer Decoder Stack**
- Pure decoder-only autoregressive architecture for next-token prediction
- Multi-head self-attention with 16 heads for parallel attention patterns
- Sinusoidal positional embeddings encode token sequence positions
- Pre-norm layer normalization architecture with residual connections
- Dropout regularization at multiple layers prevents overfitting

**Tokenization System**
- Byte Pair Encoding (BPE) algorithm trained from scratch on your corpus
- Configurable vocabulary size (default: 32,768 tokens)
- Special tokens: `<|bos|>` (beginning), `<|eos|>` (end), `<|pad|>` (padding)
- Full UTF-8 Unicode support for multilingual text
- Source code adapted from [Nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy

**Data Processing Pipeline**
- Explicit index-based shuffling ensures reproducibility across runs
- Binary `.bin` memory-mapped files enable instant data access without loading into RAM
- Automatic 99/1 train/validation split with separate output files
- Chunked processing handles datasets larger than available memory
- DataTrove `.ds` format uses 2-byte token encoding supporting vocabularies up to 65K

***

## 🚀 Training Pipeline

### Optimization Strategy

**Mixed Precision Training**
- Forward pass and loss computation in FP16 for 2x speed improvement
- FP32 master weights maintained for numerical stability
- Automatic gradient scaling prevents tiny gradient underflow in FP16
- Dynamic scale factor adjusts to avoid overflow (halves on overflow, doubles after 2K successful steps)
- Gradient scaler handles: loss multiplication → backprop → unscaling → overflow detection

**Gradient Accumulation**
- Simulates large batch size (32) on limited VRAM by processing 4 micro-batches of size 8
- Loss divided by accumulation steps before backward pass for proper averaging
- Gradients accumulated across micro-batches before single optimizer step
- Enables training with effective batch sizes impossible to fit in GPU memory

**Learning Rate Schedule**
- Linear warmup over first 50 steps for early training stability
- Cosine annealing decay from 6e-4 to 6e-5 over 10,000 total steps
- Dynamic per-step adjustment tracked in training logs

**Stability Mechanisms**
- Gradient clipping with max norm 1.0 prevents exploding gradients
- Weight decay (L2 regularization) at 0.1 for parameter regularization
- Dynamic loss scaling adapts to prevent both FP16 overflow and underflow

### Data Loading Optimization

**Streaming Architecture**
- Binary memory-mapped `.ds` files provide instant random access to tokenized sequences
- Cyclic iterators using `cycle()` enable infinite streaming without epoch boundaries or StopIteration handling
- 2 worker processes with 2x prefetch factor overlap I/O with GPU computation
- Persistent workers stay alive between batches, eliminating process reload overhead
- Pinned memory enables non-blocking CUDA transfers during forward/backward passes

### Performance Features

- Fused AdamW optimizer provides kernel-level speedups over standard Adam
- Ready for `torch.compile()` when uncommented (provides ~2x additional speedup)
- Model FLOPS Utilization (MFU) tracking measures achieved FLOPS vs theoretical GPU peak
- Non-blocking CUDA transfers overlap data loading with computation

### Training Flow

```
Dynamic LR Adjustment → Gradient Accumulation (4 micro-steps) → Mixed Precision FP16
         ↓                           ↓                                    ↓
    Per-step rate            Simulate batch=32                 FP16 compute operations
                            with batch=8 memory                FP32 master parameters
         ↓                           ↓                                    ↓
  Gradient Clipping → Weight Update (AdamW) → Validation Probes (every 100 steps)
```

### Monitoring & Logging

- Real-time metrics logged every 10 steps: loss, learning rate, MFU percentage
- Validation runs every 100 steps monitor generalization and detect overfitting
- Hardware utilization tracked through Model FLOPS Utilization calculation
- Training uses autoregressive teacher forcing: P(token_t | context) with input shifted by one position

***

## ⚙️ Training Configuration Reference

| Component | Value | Purpose |
|-----------|-------|---------|
| **Batch Size** | 8 | Fits in GPU memory constraint |
| **Grad Accumulation** | 4 steps | Effective batch = 32 |
| **Learning Rate** | 6e-4 → 6e-5 | Cosine decay schedule |
| **Warmup Steps** | 50 | Early training stability |
| **Total Steps** | 10,000 | Full training run |
| **Weight Decay** | 0.1 | L2 regularization strength |
| **Gradient Clip** | 1.0 | Explosion prevention threshold |
| **Data Workers** | 2 | Parallel data loading |
| **Prefetch Factor** | 2 | Batches loaded ahead |

***

## 🎓 Implementation Highlights

**What Makes This Production-Ready**
- Complete pipeline from raw text to trained model (no missing steps)
- Efficient binary data format eliminates I/O bottlenecks during training
- Mixed precision training maximizes GPU utilization while maintaining stability
- Gradient accumulation enables training with batch sizes larger than VRAM capacity
- Cyclic data loading eliminates epoch boundaries and data reloading overhead

**Explore the Code to Understand**
- How gradient scaler prevents FP16 underflow while avoiding overflow
- Why loss is divided by accumulation steps before the backward pass
- MFU calculation methodology comparing achieved vs theoretical peak FLOPS
- How cyclic iterators enable infinite training without data reloading
- The complete mixed precision flow from forward pass to parameter update

**Inline Comments Explain**
- Full gradient scaling mechanics: scaling → backward → unscaling → update
- Loss averaging strategy across micro-batches in gradient accumulation
- Dynamic scale factor adjustment logic for mixed precision stability

***

<div align="center">

**🔬 Built for learning and experimentation with production-grade techniques 🚀**

</div>

***
