# DeepSeek R1-Based LLM Implementation

A from-scratch implementation of a language model inspired by the **DeepSeek-R1** architecture, built entirely in PyTorch. This project reproduces the key architectural innovations from the DeepSeek R1 technical report — including Multi-Head Latent Attention, Mixture of Experts with auxiliary-loss-free load balancing, Decoupled Rotary Positional Embeddings, and Multi-Token Prediction — and trains them on the Tiny Shakespeare dataset as a compact educational example.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Components](#key-components)
  - [Input Embeddings](#input-embeddings)
  - [Decoupled Rotary Positional Embeddings (RoPE)](#decoupled-rotary-positional-embeddings-rope)
  - [Multi-Head Latent Attention (MLA)](#multi-head-latent-attention-mla)
  - [Mixture of Experts (MoE)](#mixture-of-experts-moe)
  - [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
  - [RMSNorm](#rmsnorm)
- [Theoretical References](#theoretical-references)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Training](#training)
- [Inference](#inference)
- [Dependencies](#dependencies)

---

## Architecture Overview

The model follows a decoder-only transformer design with the following pipeline:

```
Input Token IDs
      │
      ▼
 Input Embeddings (scaled by √d_model)
      │
      ▼
 Embedding Dropout
      │
      ▼
 ┌──────────────────────────────────┐
 │     Transformer Block (×N)       │
 │                                  │
 │  x ─► RMSNorm ─► MLA ─► + x     │  (attention sub-layer with residual)
 │  x ─► RMSNorm ─► MoE MLP ─► + x │  (feed-forward sub-layer with residual)
 │                                  │
 └──────────────────────────────────┘
      │
      ▼
 Final RMSNorm
      │
      ├─► LM Head ──► Next-Token Logits
      │
      └─► MTP Modules (×K) ──► Future-Token Logits (training only)
```

---

## Key Components

### Input Embeddings

Converts discrete token IDs into dense continuous vectors via a learned lookup table (`nn.Embedding`). An optional scaling factor of `√d_model` is applied to stabilize training, following the original Transformer convention.

```
input_ids [B, T]  →  nn.Embedding  →  [B, T, d_model] * √d_model
```

### Decoupled Rotary Positional Embeddings (RoPE)

Based on the **decoupled RoPE** strategy from DeepSeek-R1, positional information is injected into the attention mechanism rather than added to the embeddings directly. This design has two key properties:

- **Rotation-based encoding**: Each pair of dimensions in the query/key vectors is treated as a 2D vector rotated by a position-dependent angle θ. The rotation formula is:
  ```
  x' = x · cos(θ) + rotate_half(x) · sin(θ)
  ```
  where `rotate_half` splits the vector into two halves `[x1, x2]` and returns `[-x2, x1]`.

- **Decoupled design**: RoPE is applied only to a dedicated subset of the query/key dimensions (`rope_head_dim`), not the full head dimension. The queries get per-head RoPE components while the keys share a **single** RoPE projection across all heads, making position encoding memory-efficient. The content (non-positional) and RoPE parts are concatenated before computing attention scores:
  ```
  q = [q_content, q_rope]     k = [k_content, k_rope_shared]
  ```

This decoupling allows the KV cache to store only the low-rank latent vector plus the shared RoPE key, significantly reducing memory during inference.

### Multi-Head Latent Attention (MLA)

The core attention mechanism from DeepSeek-R1 that achieves KV cache compression through **low-rank latent projections**:

```
          x
          ├── Q path:  x → W_dq (down-project to q_lora_rank)
          │                 ├─► W_uq → q_content  [B, H, T, head_dim]
          │                 └─► W_qr → q_rope     [B, H, T, rope_head_dim]
          │
          └── KV path: x → W_dkv (down-project to kv_lora_rank)
                            ├─► W_uk → k_content  [B, H, T, head_dim]
                            └─► W_uv → v          [B, H, T, head_dim]
                       x → W_kr → k_rope_shared   [B, 1, T, rope_head_dim]
```

**Key ideas:**
- Queries and keys/values are first compressed to a low-rank latent space (`q_lora_rank`, `kv_lora_rank`) before being projected up to the full multi-head dimension. This reduces the parameter count and KV cache footprint.
- The shared RoPE key (`k_rope_shared`) is computed once and broadcast to all heads via `.expand()`, avoiding redundant computation.
- Attention is computed over the concatenated `[content, rope]` dimensions with a scaling factor of `1/√(head_dim + rope_head_dim)`.
- Causal masking is applied to prevent attending to future tokens.

### Mixture of Experts (MoE)

The feed-forward sub-layer uses a **Mixture of Experts** architecture combining always-on shared experts with dynamically routed experts:

#### Expert Architecture — SwiGLU FFN
Each expert is a SwiGLU feed-forward network:
```
output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

#### Shared Experts
A fixed set of experts that process **every** token. Their outputs are summed and always contribute to the final result, providing a stable baseline representation.

#### Routed Experts with Auxiliary-Loss-Free Balancing
A sigmoid-based router selects the top-k experts for each token. The key innovation from DeepSeek-R1 is the **loss-free bias balancing** mechanism:

1. **Affinity scores**: `sigmoid(router_linear(x))` — per-token scores for each expert.
2. **Biased selection**: `affinity + expert_bias` — a learned bias nudges expert selection toward balanced load.
3. **Top-k selection**: Experts are chosen based on biased scores, but the **final mixing weights** are computed from the original (unbiased) affinity scores, normalized to sum to 1.
4. **Bias update** (after each training step):
   ```
   bias_i ← bias_i - γ · sign(load_i - mean_load)
   ```
   - Overloaded expert → bias decreases → less likely to be selected
   - Underloaded expert → bias increases → more likely to be selected

This eliminates the need for auxiliary balancing losses that can interfere with the main training objective.

#### Combined MoE Output
```
output = shared_experts(x) + Σ (gate_i · routed_expert_i(x))   for top-k experts
```

### Multi-Token Prediction (MTP)

During training, the model predicts not only the next token but also **future tokens** at multiple depths, following the MTP approach from the DeepSeek-R1 paper.

Each MTP module at depth `k` works as follows:

```
h'_k = Linear( [RMSNorm(prev_hidden), RMSNorm(Embed(future_token))] )
h_k  = TransformerBlock(h'_k)
logits_k = LM_Head(FinalNorm(h_k))
```

- **Depth 1** predicts token `t+2` (the token after next)
- **Depth 2** predicts token `t+3`, and so on
- Each depth receives the hidden state from the previous depth (chained sequentially)
- The MTP loss is averaged across all depths and scaled by `mtp_lambda`

**Total training loss:**
```
loss = cross_entropy(main) + λ · mean(cross_entropy(mtp_depth_1), ..., cross_entropy(mtp_depth_K))
```

MTP modules share the same embedding layer and LM head as the main model. During inference, MTP modules can optionally be used for **speculative decoding** (drafting multiple future tokens in parallel).

### RMSNorm

Root Mean Square Layer Normalization is used throughout the model instead of standard LayerNorm. Unlike LayerNorm, RMSNorm does **not** subtract the mean — it only rescales by the root mean square:

```
x̂_i = x_i / √(mean(x²) + ε)
output = x̂ * weight
```

This is computationally cheaper and is the normalization used by LLaMA, DeepSeek, and other modern LLMs.

---

## Theoretical References

The implementation is grounded in theoretical notes located in the `notes/` directory. These documents serve as the primary reference material for each architectural component:

| Document | Topic | Relevant Component |
|---|---|---|
| `Deepseek r1 multi head latent attention.pdf` | Multi-Head Latent Attention with low-rank KV compression and decoupled RoPE integration | `MultiHeadLatentAttention` class |
| `R1 positional encoding.pdf` | Decoupled Rotary Positional Embeddings — how RoPE is separated from content projections and shared across heads | `RotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb` |
| `R1 mixture of experts.pdf` | Mixture of Experts architecture with auxiliary-loss-free bias balancing, shared + routed experts, and SwiGLU FFN | `MoEMLP`, `LossFreeBiasRouter`, `SwiGLUExpert`, `SharedExperts` |
| `Multi-token prediction.pdf` | Multi-Token Prediction training objective — chained depth modules for predicting multiple future tokens | `MTPModule`, `_forward_mtp` |
| `R1 Quantization.pdf` | Quantization techniques for efficient inference of large-scale models | Background knowledge for deployment considerations |

---

## Project Structure

```
DeepSeek R1 Based LLM/
│
├── r1_based_llm_implementation.ipynb   # Main implementation notebook
│
├── data/
│   ├── input.txt                       # Tiny Shakespeare dataset
│   └── bpe_tokenizer.json              # Trained BPE tokenizer
│
├── models/
│   └── config.json                     # Model hyperparameter configuration
│
├── notes/                              # Theoretical reference documents
│   ├── Deepseek r1 multi head latent attention.pdf
│   ├── Multi-token prediction.pdf
│   ├── R1 mixture of experts.pdf
│   ├── R1 positional encoding.pdf
│   └── R1 Quantization.pdf
│
├── saved/                              # Training artifacts (plots, history)
│
└── README.md
```

---

## Configuration

The model is configured via the `ModelConfig` dataclass. Default configuration used for training:

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 1056 | BPE vocabulary size (≈√dataset_length) |
| `max_seq_len` | 128 | Maximum sequence length |
| `d_model` | 128 | Model hidden dimension |
| `num_layers` | 2 | Number of transformer blocks |
| `num_heads` | 4 | Number of attention heads |
| `head_dim` | 32 | Dimension per attention head (content) |
| `q_lora_rank` | 32 | Query low-rank compression rank |
| `kv_lora_rank` | 32 | Key/Value low-rank compression rank |
| `rope_head_dim` | 16 | RoPE dimension per head |
| `mlp_hidden_dim` | 256 | FFN expert hidden dimension |
| `num_shared_experts` | 1 | Always-on shared experts |
| `num_routed_experts` | 4 | Dynamically routed experts |
| `top_k` | 2 | Experts selected per token |
| `mtp_depth` | 2 | Number of MTP prediction depths |
| `mtp_lambda` | 1.0 | MTP loss scaling factor |
| `tie_word_embeddings` | true | Share weights between embedding and LM head |

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Installation

```bash
pip install torch tokenizers scipy matplotlib
```

### Quick Start

Open `r1_based_llm_implementation.ipynb` in Jupyter / PyCharm and run all cells sequentially. The notebook will:

1. Download the Tiny Shakespeare dataset
2. Train a BPE tokenizer
3. Build the model with all R1-inspired components
4. Train the model
5. Generate text from a prompt

---

## Training

The training loop includes:

- **AdamW optimizer** with weight decay and gradient clipping
- **Checkpoint saving**: best model and last checkpoint are saved to `models/`
- **Router bias update**: called after each optimizer step to maintain expert load balance
- **Training curves**: loss plots saved to `saved/training_curve.png`
- **MTP training**: multi-token prediction loss is computed alongside the main next-token loss

Training can be resumed from the last checkpoint automatically.

---

## Inference

Two inference utilities are provided:

- **`generate_text_with_mtp`**: Autoregressive text generation with optional sampling (temperature, greedy) and MTP-based speculative decoding support.
- **`inspect_mtp_predictions`**: Diagnostic tool that shows the model's main prediction and MTP draft predictions for a given prompt.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `tokenizers` | HuggingFace BPE tokenizer training and inference |
| `scipy` | Smoothing utilities for training curves |
| `matplotlib` | Training loss visualization |

