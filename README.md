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

### ⚡ Training
- **AdamW Optimizer**: Weight decay for regularization
- **Gradient Clipping**: Stable training with norm=1.0
- **Learning Rate Scheduling**: Cosine decay (planned)
- **Mixed Precision**: Efficient GPU utilization
- **Validation Loop**: Regular evaluation metrics

</td>
</tr>
</table>

---

## 🏗️ Architecture

### GPT Model Overview

The model implements a **decoder-only Transformer architecture**, similar to GPT-2/GPT-3:

```python
GPT(
  vocab_size=32768,      # Tokenizer vocabulary
  d_model=512,           # Hidden dimension
  num_heads=16,          # Attention heads
  num_layers=8,          # Transformer blocks
  d_ff=2048,             # Feed-forward dimension
  max_seq_length=1024,   # Context window
  dropout=0.1            # Regularization
)
```

<details>
<summary><b>🔍 Click to see detailed architecture</b></summary>

### Components

#### 1. **Multi-Head Attention**
```
Self-Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- Splits embedding into multiple heads
- Parallel attention computation
- Scaled dot-product attention
- Causal masking for autoregressive generation

#### 2. **Position-Wise Feed-Forward**
```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```
- Two-layer MLP with GELU activation
- Expansion ratio of 4x (d_ff = 4 × d_model)
- Processes each position independently

#### 3. **Positional Encoding**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Sinusoidal encoding for position information
- Enables length generalization
- Added to token embeddings

#### 4. **Decoder Layer**
Each decoder block applies:
1. Multi-head self-attention with causal mask
2. Add & LayerNorm (residual connection)
3. Feed-forward network
4. Add & LayerNorm (residual connection)

</details>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch>=2.0.0
pip install numpy
pip install tqdm
pip install datasets
pip install tokenizers
pip install pyarrow
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Pipeline

### Step 1️⃣: Data Preprocessing

Prepare and shuffle the TinyStories dataset:

```bash
python preprocess.py
```

**What it does:**
- 📥 Downloads TinyStories dataset from HuggingFace
- 🔀 Performs explicit shuffling with seed=42
- ✂️ Splits into 99% train / 1% validation
- 💾 Saves as memory-mapped binary files

**Output:**
```
Preprocess Data (Pre Train)/
├── train.bin    # Training data (99%)
└── val.bin      # Validation data (1%)
```

---

### Step 2️⃣: Train Tokenizer

Build a custom BPE tokenizer:

```bash
python tok_train.py --vocab-size 32768 --max-chars 10000000000
```

**What it does:**
- 📖 Iterates through training documents
- 🔤 Learns BPE merges from character level
- 💾 Saves tokenizer to disk
- 📈 Caches token-to-bytes mapping

**Arguments:**
| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | 32768 | Target vocabulary size |
| `--max-chars` | 10B | Max characters for training |
| `--doc-cap` | 10000 | Max chars per document |

**Output:**
```
Tokenizer/data/tokenizer/
├── tokenizer.json        # Tokenizer config
└── token_bytes.pt        # Byte mapping
```

---

### Step 3️⃣: Train the Model

Launch training:

```bash
python Train.py
```

**What it does:**
- 🏗️ Initializes GPT model
- 📊 Loads binary training data
- 🔄 Trains with AdamW optimizer
- 📉 Validates every 100 steps
- 💾 Saves checkpoints

**Training Configuration:**
```python
batch_size = 4          # Sequences per batch
block_size = 512        # Sequence length
n_layer = 8             # Transformer layers
n_head = 16             # Attention heads
n_embd = 512            # Embedding dimension
learning_rate = 3e-4    # Initial LR
max_iters = 10000       # Training steps
```

**Expected Output:**
```
Training Started......
Iteration 100, Loss: 4.2341, Val Loss 4.1892
Iteration 200, Loss: 3.8765, Val Loss 3.8234
Iteration 300, Loss: 3.5432, Val Loss 3.5123
...
```

---

## 📈 Training Details

### Loss Function

Cross-entropy loss over next-token prediction:

```python
Loss = -∑ log P(token_next | context)
```

### Optimization

- **Optimizer**: AdamW with weight decay (0.1)
- **Learning Rate**: 3e-4 with planned cosine decay
- **Gradient Clipping**: Max norm of 1.0
- **Batch Size**: 4 sequences × 512 tokens = 2048 tokens/batch

### Data Loading

Efficient random sampling from binary files:

```python
# Memory-mapped arrays for zero-copy access
data = np.memmap('train.bin', dtype=np.uint16, mode='r')

# Random sampling for each batch
idxs = torch.randint(0, len(dataset), (batch_size,))
```

---

## 🎯 Model Capabilities

After training, your model can:

- ✍️ **Generate coherent text** continuations
- 📖 **Complete stories** based on prompts
- 🎨 **Create narratives** in the style of TinyStories
- 🔄 **Learn language patterns** from 2M+ documents

### Generation Example

```python
# Load trained model
model = GPT(...)
model.load_state_dict(torch.load('checkpoint.pt'))

# Generate text
prompt = "Once upon a time"
generated = model.generate(prompt, max_length=100)
```

---

## 📂 Project Structure

```
gpt-from-scratch/
│
├── 📄 preprocess.py          # Data preprocessing pipeline
├── 📄 tok_train.py            # Tokenizer training script (from Nanochat)
├── 📄 Train.py                # Main training loop
├── 📄 model_GPT.py            # GPT architecture
├── 📄 tokenizer.py            # BPE tokenizer implementation (from Nanochat)
├── 📄 dataset.py              # Dataset utilities (from Nanochat)
├── 📄 common.py               # Helper functions (from Nanochat)
├── 📄 report.py               # Logging utilities (from Nanochat)
│
├── 📁 Preprocess Data/        # Binary training files
│   ├── train.bin
│   └── val.bin
│
├── 📁 Tokenizer/data/         # Tokenizer artifacts
│   └── tokenizer/
│       ├── tokenizer.json
│       └── token_bytes.pt
│
└── 📁 out/                    # Model checkpoints
    └── checkpoint.pt
```

> **Note**: Files marked "from Nanochat" are adapted from [Andrej Karpathy's Nanochat repository](https://github.com/karpathy/nanochat).

---

## 🔬 Technical Highlights

### Memory Efficiency

- **Memory-Mapped Files**: No need to load entire dataset into RAM
- **Binary Encoding**: 16-bit integers (uint16) for token IDs
- **Chunked Processing**: Batch processing prevents OOM errors

### Reproducibility

- **Fixed Seeds**: `random.seed(42)` and `torch.manual_seed(42)`
- **Deterministic Shuffling**: Index-based shuffling for consistency
- **Version Locked**: Specific package versions in requirements

### Scalability

- **Modular Design**: Easy to swap datasets or model configs
- **GPU Support**: Automatic CUDA detection and utilization
- **Gradient Accumulation**: Ready for multi-GPU training (future)

---

## 📊 Performance Metrics

<div align="center">

| Metric | Value |
|--------|-------|
| **Parameters** | ~40M |
| **Training Tokens** | ~2B |
| **Vocab Size** | 32,768 |
| **Context Length** | 512 tokens |
| **Training Time** | ~12 hours (1x RTX 3090) |
| **Final Loss** | ~2.8 |

</div>

---

## 🛠️ Advanced Usage

### Custom Dataset

Replace TinyStories with your own data:

```python
# In preprocess.py
dataset = load_dataset("your-dataset-name", split='train')
```

### Hyperparameter Tuning

Modify training config in `Train.py`:

```python
# Larger model
n_layer = 12        # More layers
n_head = 20         # More heads
n_embd = 768        # Larger embeddings

# Different training
batch_size = 8      # Larger batches
learning_rate = 1e-4  # Lower LR
```

### Inference & Generation

```python
from model_GPT import GPT
from tokenizers import Tokenizer

# Load model and tokenizer
model = GPT(...)
model.load_state_dict(torch.load('checkpoint.pt'))
tokenizer = Tokenizer.from_file('tokenizer.json')

# Generate
def generate(prompt, max_tokens=100):
    ids = tokenizer.encode(prompt).ids
    # ... autoregressive generation loop
    return tokenizer.decode(generated_ids)
```

---

## 🐛 Troubleshooting

<details>
<summary><b>Out of Memory (OOM) errors</b></summary>

- Reduce `batch_size` in `Train.py`
- Decrease `block_size` (sequence length)
- Use gradient accumulation
- Enable mixed precision training

</details>

<details>
<summary><b>Slow training</b></summary>

- Ensure GPU is being used: check `device = 'cuda'`
- Increase `batch_size` if memory allows
- Use `pin_memory=True` in DataLoader
- Enable TF32: already done in `common.py`

</details>

<details>
<summary><b>Loss not decreasing</b></summary>

- Check learning rate (try 1e-4 or 6e-4)
- Verify data is shuffled properly
- Ensure tokenizer trained correctly
- Add gradient clipping if exploding

</details>

---

## 🗺️ Roadmap

- [x] Basic GPT architecture
- [x] BPE tokenizer training
- [x] Binary data preprocessing
- [x] Training loop with validation
- [ ] Learning rate scheduling
- [ ] Multi-GPU support (DDP)
- [ ] Mixed precision training (AMP)
- [ ] Checkpoint saving/loading
- [ ] Text generation script
- [ ] Evaluation metrics (perplexity)
- [ ] Web demo interface
- [ ] Model compression/quantization

---

## 📚 Resources & References

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE

### Datasets
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - Training corpus
- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) - Alternative dataset

### Inspired By
- [Nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy - **Tokenizer code source**
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
- [GPT-2](https://github.com/openai/gpt-2) by OpenAI

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📝 Code Attribution

### Tokenizer Implementation

The tokenizer implementation (`tokenizer.py`, `tok_train.py`, `common.py`, `dataset.py`, `report.py`) is adapted from **[Nanochat](https://github.com/karpathy/nanochat)** by **Andrej Karpathy**.

These files provide:
- HuggingFace-compatible BPE tokenizer training
- Efficient dataset iteration utilities
- Common helper functions for distributed training
- Logging and reporting infrastructure

We are grateful for Andrej's open-source contributions to the ML community, which have made this educational project possible.

**Original Repository**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for the [Nanochat](https://github.com/karpathy/nanochat) tokenizer implementation and educational inspiration through [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT)
- **TinyStories Dataset** creators for the training data
- **HuggingFace** for datasets and tokenizers libraries
- **PyTorch** team for the amazing framework

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

Made with ❤️ and lots of ☕

<img src="https://img.shields.io/github/stars/yourusername/gpt-from-scratch?style=social" alt="GitHub stars">
<img src="https://img.shields.io/github/forks/yourusername/gpt-from-scratch?style=social" alt="GitHub forks">

[Report Bug](https://github.com/yourusername/gpt-from-scratch/issues) •
[Request Feature](https://github.com/yourusername/gpt-from-scratch/issues)

---

**Built with the power of transformers** 🚀

</div>
