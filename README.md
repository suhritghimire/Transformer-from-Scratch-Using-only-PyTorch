# 🔁 Transformer from Scratch (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade implementation of the original **"Attention Is All You Need"** Transformer architecture (Vaswani et al., 2017), built from scratch using **PyTorch**. This project demonstrates the full pipeline of Neural Machine Translation (NMT), from dataset processing to model training and inference.

## 🌐 English-to-Urdu Translation
This model is specifically configured for **English-to-Urdu** translation, leveraging the **OPUS-100** dataset. It handles the complexities of Urdu's Perso-Arabic script through custom subword tokenization and optimized training loops.

---

## ✨ Features

- **Full Architecture**: Implementation of Multi-Head Attention, Positional Encoding, and Feed-Forward networks.
- **Advanced Decoding**: Support for both Greedy decoding and optimized **Beam Search**.
- **Hardware Acceleration**: Out-of-the-box support for **MPS (Metal Performance Shaders)** on Mac and CUDA for NVIDIA GPUs.
- **Monitoring**: Integrated with **TensorBoard** for real-time loss and metric tracking.
- **Expert Validation**: Includes attention visualization notebooks to audit model focus.

---

## 📁 Project Structure

```bash
transformer-from-scratch/
├── model.py          # Core Transformer architecture (Encoder/Decoder)
├── dataset.py        # Bilingual dataset loading and tokenization logic
├── train.py          # Training loop with validation and checkpointing
├── config.py         # Global project and model hyperparameters
├── translate.py      # Command-line interface for translation inference
├── Beam_Search.ipynb # Interactive exploration of Beam Search decoding
└── attention_visual.ipynb  # Visual audit of self-attention heatmaps
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/suhritghimire/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

### 2. Training
The model will automatically download the OPUS-100 dataset and build the tokenizers on first run.
```bash
python3 train.py
```

### 3. Inference
```bash
python3 translate.py "Hello, how are you?"
```

---

## 📊 Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **d_model** | 512 | Model dimensionality |
| **Heads** | 8 | Number of attention heads |
| **Layers** | 6 | Encoder and Decoder depth |
| **d_ff** | 2048 | Feed-forward network dimension |
| **Dropout** | 0.1 | Regularization rate |
| **Source** | English | Input Language |
| **Target** | Urdu | Output Language |

---

## 📚 References

> Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---
© 2025 Suhrit Ghimire. Licensed under [MIT](LICENSE).
