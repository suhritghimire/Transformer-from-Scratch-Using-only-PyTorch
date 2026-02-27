# 🔁 Transformer from Scratch (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, research-grade implementation of the original **"Attention Is All You Need"** Transformer architecture (Vaswani et al., 2017) — built from scratch using only **PyTorch**, without relying on any higher-level libraries.

Trained on an **English–Urdu** parallel corpus as a neural machine translation (NMT) task.

---

## ✨ Features

- Multi-head self-attention and cross-attention
- Positional encoding (sinusoidal)
- Label smoothing loss
- Beam search decoding
- Attention visualization
- TensorBoard + Weights & Biases training tracking

---

## 📁 Structure

```
transformer-from-scratch/
├── model.py          # Full Transformer (encoder, decoder, MHA, FFN)
├── dataset.py        # Tokenization & DataLoader
├── train.py          # Training loop
├── config.py         # Hyperparameters
├── translate.py      # Inference script
├── Beam_Search.ipynb # Beam search exploration
├── attention_visual.ipynb  # Attention heatmaps
└── requirements.txt
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python train.py
```

For a Colab-based run: `Colab_Train.ipynb`

---

## 📊 Architecture

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| Heads | 8 |
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| FFN dim | 2048 |
| Source Lang | English |
| Target Lang | Urdu |

---

## 📚 Reference

> Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
