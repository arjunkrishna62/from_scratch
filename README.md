# LLM From Scratch

A simplified implementation of the **Transformer** architecture ("Attention Is All You Need") from scratch in Python using **PyTorch**, built as part of my Master’s research projects.

## Overview

This project implements:

- **Multi-head self-attention**
- **Position-wise feed-forward networks**
- **Sinusoidal positional encodings**
- **Encoder and decoder stacks**
- **Feed Forward(MLP)**

It trains a Transformer model for **language modeling** and **classification/instruction-following** tasks — all without using high‑level libraries like Hugging Face.

## Motivation

- Reinforce understanding of core Transformer components (attention, residuals, layer norm).
- Learn how training loops, batching, masking, and decoding work under the hood.
- Bridge theory and code by building from the ground up.

## Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/arjunkrishna62/from_scratch.git
cd from_scratch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

from_scratch/
├── data/                # Dataset loader & preprocessing
├── model.py             # Transformer architecture
├── train.py             # Training loop with batching and optim
├── inference.py         # Beam search / sample decoder
├── utils.py             # Helper functions (tokenization, evaluation)
└── README.md


python train.py \
  --dataset data/sample.txt \
  --epochs 10 \
  --batch_size 32 \
  --d_model 128 \
  --num_heads 8 \
  --num_layers 4


python inference.py \
  --checkpoint checkpoints/epoch_10.pt \
  --prompt "Translate to French: Hello, how are you?" \
  --max_len 50


---

Let me know if you'd like to add performance stats, data samples, or badges for frameworks/framework versions!

