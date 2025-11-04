# Mixture of Experts (MoE) — PyTorch Implementation

[![PyTorch](https://img.shields.io/badge/Built_with-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/Inspired_by-DeepSeekV3-black?logo=openai)](https://github.com/deepseek-ai)

### Author

**Implementation by:** Keeratpreet Singh

**Background:** Self-taught 16-year-old developer passionate about AI systems, LLM architectures, and efficient computation in deep learning.

**Concept originally from:**
Mixture of Experts (MoE) mechanism explored in
Switch Transformers (Google Research, 2021)
,
GShard (2020)
,
and Sparsely-Gated MoE (2017)

**Framework:** PyTorch
**Language:** Python

## Disclaimer

This is an independent educational implementation of the Mixture of Experts (MoE) architecture used in large-scale transformer systems such as GShard, Switch Transformer, and DeepSeek-V3.
I am not affiliated with Google, DeepSeek, or any related institution.
This project is provided solely for learning, research, and experimentation.

## Overview

This repository contains a clean and readable PyTorch implementation of the Mixture of Experts (MoE) layer — a modular technique that enables conditional computation in neural networks.

Instead of processing every token through the same feed-forward network, MoE routes each token to a small subset of experts (e.g., top-2), drastically improving model capacity and efficiency.

Two simplified yet practical MoE implementations are provided:

Basic MoE (Deterministic Routing)

Noisy MoE (Stochastic Routing + Load Balancing)

## Key Features

**Top-2 Expert Routing:**
Each token is dynamically assigned to its top-2 experts using a gating network.

**SwigLU Expert Network:**
Each expert is a lightweight feed-forward module with the Swish-Gated Linear Unit (SwigLU) activation.

**Noise Injection for Routing Diversity:**
The improved MoE version introduces Gaussian-like random noise to encourage balanced expert utilization.

**Load Balancing Loss:**
Penalizes routing imbalance to prevent certain experts from dominating.

**Dropout Regularization:**
Applied on gating logits to improve generalization during training.

## Code Structure
### SwigLU Expert (swigluffn)

A compact feed-forward expert with a SwigLU activation mechanism:

class swigluffn(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.w1 = nn.Linear(d, dff)
        self.w2 = nn.Linear(d, dff)
        self.w3 = nn.Linear(dff, d)

    def forward(self, x):
        return self.w3(self.w2(x) * nn.functional.silu(self.w1(x)))

### Basic MoE (code1) — Deterministic Routing

Uses top-2 experts per token

Computes weighted combination using softmax-normalized routing priorities

No regularization or balancing (ideal for conceptual understanding)

moe = MOE(noe=8, contextlenght=128, cfg={"emb_dim": 512, "dff": 2048})
out = moe(x)  # returns [batch, seq_len, emb_dim]

### Improved MoE (code2) — Noisy Routing + Balancing

Adds stochastic noise to the gating logits

Computes load balancing loss to distribute tokens fairly

Supports dropout and training vs inference modes

moe = MOE(num_experts=8, context_length=128, cfg={
    "emb_dim": 512,
    "dff": 2048,
    "dropout_rate": 0.1
})

x = torch.randn(4, 128, 512)
out, loss = moe(x)  # during training

### Forward Pass Summary

1. Gating:
Each token produces a gating vector → selects top-2 experts.

2. Routing:
Tokens routed to selected experts → outputs weighted and combined.

3. Noisy Top-k (optional):
Adds random noise to encourage expert diversity.

4. Load Balancing:
Applies auxiliary loss term to ensure even expert utilization.

📘 Example Usage
import torch
from moe import MOE, swigluffn

cfg = {
    "emb_dim": 512,
    "dff": 2048,
    "dropout_rate": 0.1
}

x = torch.randn(2, 128, cfg["emb_dim"])

moe = MOE(num_experts=8, context_length=128, cfg=cfg)
out, loss = moe(x)

print(out.shape, loss.item())

## Research References

Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Google Brain, 2017)

Lepikhin et al., GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Google Research, 2020)

Fedus et al., Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Google Research, 2021)

DeepSeek AI, DeepSeek-V3: Towards Efficient Sparse-Latent LLM Architectures (2025)

## Possible Applications

Transformer / LLM feed-forward replacement

Distributed expert training research

Sparse compute experimentation

Educational demonstrations of routing mechanisms

## Citation

If you use or reference this work, please cite both the original MoE papers and this re-implementation:

@software{keeratsingh2025moe,
  author = {Keeratpreet Singh},
  title = {Mixture of Experts (Simplified PyTorch Implementation)},
  year = {2025},
  url = {https://github.com/Keeratpreetsingh}
}

💡 Contact

📧 Email: keeratpreetsingh2@gmail.com

🌐 GitHub: Keeratpreetsingh

🪪 License

This repository is released under the MIT License.
Original research © Google Research, DeepSeek AI.
Implementation © 2025 Keeratpreet Singh.
