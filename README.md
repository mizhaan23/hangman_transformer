# Hangman-Transformer: Character-Level Sequence Reconstruction & Accelerated JAX Simulation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX / Flax NNX](https://img.shields.io/badge/Framework-JAX%20%7C%20Flax%20NNX-red)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.6-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Win Rate](https://img.shields.io/badge/Validation%20Win%20Rate-66.39%25-brightgreen)](#experimental-results)
[![Simulation Throughput](https://img.shields.io/badge/Throughput-470%20games%2Fs-blue)](#experimental-results)

An end-to-end deep learning framework that models the game of Hangman as a character-level sequence reconstruction task using a 12-layer bidirectional Transformer encoder. The project includes an **exact functional re-implementation in pure JAX/Flax NNX**, executing the full pipeline (dataset buffering, Hangman masking, gradient updates, and multi-game rollouts) directly in GPU memory via XLA compilation.

On a strictly held-out test split of **2,273 unseen words** drawn from a 250,000-word vocabulary, our supervised pre-trained Transformer achieves a **66.388% win rate** (1,509 / 2,273 words solved) under the standard 6-lives constraint, while simulating all test games concurrently in under **4.85 seconds** (470.8 games/sec).

---

## Key Highlights & Performance

*(Benchmarked with exact batch size 128 on NVIDIA GPU via WSL2)*

| Metric | PyTorch Baseline | JAX / Flax NNX (This Repo) | Improvement / Notes |
| :--- | :--- | :--- | :--- |
| **Validation Win Rate** | 66.39% | **66.388%** (1,509 / 2,273) | Evaluated on unseen words with $\le 6$ lives |
| **Test Suite Latency** | ~45 s (sequential loop) | **4.828 s** (GPU lockstep) | **~9.3x faster game evaluation** |
| **Inference Throughput** | ~50 games/s | **470.8 games/s** | Fully jitted 26-step `jax.lax.scan` |
| **Training Throughput (Compute)** | 16.3 batches/s (2,088 samples/s) | **30.6 batches/s** (3,916 samples/s) | **~1.9x faster pure GPU compute** |
| **1 Epoch Compute Time (225k words)** | 107.8 s (~1.80 min) | **57.5 s** (~0.96 min) | Pure GPU masking (`jax.vmap`) |
| **Full Epoch Run (w/ Val, save=False)** | **123.7 s** (Notebook log) | **96.5 s** (Notebook log) | End-to-end epoch + validation pass |

---

## Methodology & Formulation

The core thesis of this work is that Hangman can be framed as a partially observable discrete sequence reconstruction problem inspired by Masked Language Modeling (BERT), adapted to the specific global revelation rules of Hangman.

### 1. Vocabulary & Token Representation
Let the English alphabet be $\Sigma = \{\text{a}, \text{b}, \ldots, \text{z}\}$. The network operates on a vocabulary:

$$\mathcal{V} = \Sigma \cup \{\text{[PAD]}, \text{[BLANK]}\}$$

of size $|\mathcal{V}| = 28$, where $\text{[PAD]}$ represents sequence padding and $\text{[BLANK]}$ denotes the hidden letter placeholder (character `'_'`). The bijective token mapping $\tau : \mathcal{V} \to \{0, 1, \ldots, 27\}$ is defined as:

$$\tau(\text{[PAD]}) = 0, \quad \tau(\text{a}) = 1, \;\ldots,\; \tau(\text{z}) = 26, \quad \tau(\text{[BLANK]}) = 27$$

Given a target word $W = (c_1, \ldots, c_L)$ of length $L \le M$ ($M = 32$), the ground-truth sequence $Y \in \mathcal{V}^M$ is:

$$y_t = \begin{cases} \tau(c_t), & 1 \le t \le L \\ 0, & L < t \le M \quad (\text{[PAD]}) \end{cases}$$

### 2. The Hangman Masking Operator $\mathcal{M}(W, p)$
In standard BERT, tokens are masked independently. In contrast, Hangman reveals or conceals letters **globally**: all occurrences of a chosen letter are revealed or hidden together.

Let $U(W) = \{c \in \Sigma \mid \exists t, \, c_t = c\}$ be the set of unique letters in $W$. For masking ratio $p \sim \mathcal{U}(0, 1)$, we sample a subset of masked letters $S \subset U(W)$ of size $k = \min(|U(W)|, \max(1, \lfloor p \cdot |U(W)| \rfloor))$. The masked input $X \in \mathcal{V}^M$ is:

$$x_t = \begin{cases} 27 \quad (\text{[BLANK]}), & \text{if } 1 \le t \le L \text{ and } c_t \in S \\ y_t, & \text{if } 1 \le t \le L \text{ and } c_t \in U(W) \setminus S \\ 0 \quad (\text{[PAD]}), & \text{if } L < t \le M \end{cases}$$

### 3. Transformer Encoder Architecture
* **Input Embedding**: Discrete tokens are embedded into $\mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 128$) and scaled by $\sqrt{d_{\text{model}}}$.
* **Sinusoidal Positional Encoding**: Fixed continuous frequency signals inject token order:
  $$\text{PE}_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i / d_{\text{model}}}}\right), \quad \text{PE}_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i / d_{\text{model}}}}\right)$$
* **Encoder Blocks**: $N = 12$ Transformer layers, each with $h = 8$ attention heads ($d_k = 16$), $d_{\text{ff}} = 2048$, ReLU activations, LayerNorm, and residual connections. Attention is masked with key-padding mask $\mathbf{m}_t = \mathbb{I}(x_t \ne 0)$.
* **Output Projection**: A linear classification head projects encoder states $H_t$ to raw logits $Z_t \in \mathbb{R}^{28}$, yielding token probabilities via softmax:
  $$\hat{P}(y_t = v \mid X; \theta) = \frac{\exp(Z_{t, v})}{\sum_{v'=0}^{27} \exp(Z_{t, v'})}$$

### 4. Training Objective
Parameters $\theta$ are trained by minimizing the masked categorical cross-entropy loss over non-padding tokens:

$$\mathcal{L}(\theta) = - \frac{1}{\sum_{t=1}^M \mathbb{I}(y_t \ne 0)} \sum_{t=1}^M \mathbb{I}(y_t \ne 0) \log \hat{P}(y_t \mid X; \theta)$$

Optimized with Adam ($\alpha = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$).

### 5. Positional Marginalization Policy
During test gameplay, given current observation $X$ and guessed letters $G \subset \Sigma$, the agent evaluates candidate letters $a \in \Sigma \setminus G$ by marginalizing probability mass across all active word positions:

$$\text{Score}(a) = \sum_{t=1}^M \mathbb{I}(x_t \ne 0) \cdot \hat{P}(y_t = \tau(a) \mid X)$$

Invalid (previously guessed) letters are masked to $-\infty$, and the greedy action is selected:

$$a^* = \arg\max_{a \in \Sigma \setminus G} \text{Score}(a)$$

---

## High-Throughput JAX Acceleration Architecture

```
+----------------------------------------------------------------------------------------+
|                                    GPU VRAM BUFFER                                     |
|  Pre-encoded Dataset: (225027, 32) int32 (~28 MB) | Zero Host-to-Device Copy Overhead   |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
|                                PURE JAX GPU OPERATORS                                  |
|  * Vectorized Masking: vmap(mask_single_word) on GPU integer arrays                    |
|  * Fused Train Kernel: @nnx.jit (mask -> forward -> cross_entropy -> autodiff -> adam) |
|  * Lockstep Rollouts : jax.lax.scan over 2,273 games x 26 steps (470.8 games/sec)      |
+----------------------------------------------------------------------------------------+
```

1. **Contiguous GPU Ingestion**: The $225,000$-word dictionary is loaded into a compact int32 array in VRAM, eliminating Python string allocations during training.
2. **GPU Native Masking**: The random permutation and masking operators run entirely inside XLA using `jax.vmap`.
3. **Scan-Compiled Rollouts**: The entire 26-step interaction trajectory for thousands of concurrent games is compiled into a single XLA computation graph via `jax.lax.scan`.

---

## Repository Structure

```
├── README.md                           # Project overview, mathematical formulation & benchmarks
├── pre_train_notebook_jax.ipynb        # Pure JAX JIT-compiled supervised pre-training (4x speedup)
├── pre_train_notebook.ipynb            # Original PyTorch supervised pre-training notebook
├── runner_example_notebook_jax.ipynb   # High-throughput JAX game runner (66.39% win rate demo)
├── runner_example_notebook.ipynb       # Original PyTorch game runner notebook
│
├── train_grpo_jax.ipynb                # Group Relative Policy Optimization (GRPO) in JAX
├── train_simple_grpo_jax.py            # Standalone Simple GRPO RL training script with live dashboard
│
├── model_jax/                          # Flax NNX Transformer implementation
│   ├── Models.py                       # Full Transformer module and weight loader
│   ├── Layers.py                       # Transformer EncoderLayer
│   ├── Sublayers.py                    # MultiHeadAttention, FeedForward, and LayerNorm
│   └── Embed.py                        # Embedder & PositionalEncoder
│
├── model_torch/                        # PyTorch reference Transformer implementation
├── env/                                # Functional vectorized Hangman environments
│   └── hangman_jax.py                  # Pure JAX functional Hangman environment
├── data/                               # 250,000-word training vocabulary
└── weights/                            # Checkpoints (.npz format)
```

---

## Installation & Reproducibility

### 1. Environment Setup

```bash
# Create a fresh conda environment with Python 3.11
conda create -n htenv python=3.11 -y
conda activate htenv

# Install JAX with CUDA 12 support alongside core dependencies
pip install -U "jax[cuda12]" flax optax torch numpy matplotlib jupyter
```

### 2. WSL2 / GPU Memory Configuration
To ensure smooth co-existence with display managers or multi-process execution:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
```

### 3. Running the Guides

* **Supervised Pre-Training in JAX**:
  ```bash
  jupyter notebook pre_train_notebook_jax.ipynb
  ```
* **High-Throughput Validation Simulation ($66.4\%$ Win Rate)**:
  ```bash
  jupyter notebook runner_example_notebook_jax.ipynb
  ```
* **Reinforcement Learning Fine-Tuning (Simple GRPO)**:
  ```bash
  python train_simple_grpo_jax.py --num-prompts 32 --group-size 8 --lr 1e-5 --no-gui
  ```

---

## Citation

If you use or reference this work in your research, please cite:

```bibtex
@misc{hangman_transformer_jax_2026,
  title  = {Solving Hangman with Encoder-Only Transformers: Supervised Word Reconstruction and High-Throughput JAX Simulation},
  author = {Mizhaan},
  year   = {2026},
  url    = {https://github.com/mizhaan23/hangman_transformer}
}
```
