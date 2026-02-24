# Transformer From Scratch: English to Sanskrit Translation 

Designed and Implemented by **Suhrit Ghimire**

Welcome to a pure PyTorch implementation of the Transformer architecture, built entirely from the ground up. This project demonstrates a deep understanding of the landmark "Attention Is All You Need" paper, applied to the sophisticated task of translating English to Sanskrit.

---

## 🚀 Overview

This repository contains a complete, robust, and highly documented implementation of the Transformer model. Unlike implementations that rely on high-level libraries, this project builds every component—from Multi-Head Attention to Positional Encoding—using only fundamental PyTorch tensors and modules.

### Key Features
- **Pure PyTorch Implementation**: No high-level abstractions; understand every matrix multiplication.
- **English-to-Sanskrit Translation**: Leveraging the modern `Saamayik` dataset for high-quality contemporary translation.
- **Custom Tokenization**: Optimized Word-Level tokenizers for both English and Sanskrit.
- **Visual Attention**: Built-in mechanisms for visualizing attention maps (see `attention_visual.ipynb`).
- **Comprehensive Training Pipeline**: Includes persistence, validation metrics (BLEU, WER, CER), and Tensorboard integration.

---

## 🛠️ Architecture Deep Dive

The architecture follows the original Transformer design exactly:

### 1. Encoder Stack
- **Input Embedding**: Vector representation of tokens.
- **Positional Encoding**: Injecting sequence order information using sine and cosine functions.
- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces.
- **Feed Forward Networks**: Position-wise fully connected layers.
- **Layer Normalization & Residual Connections**: Essential for training stability and deep networks.

### 2. Decoder Stack
- **Masked Multi-Head Attention**: Prevents positions from attending to subsequent positions (causal masking).
- **Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the input sequence.
- **Final Linear & Softmax Layer**: Projects internal representations to vocabulary space for token prediction.

---

## 📥 Dataset

We utilize the **Saamayik Dataset** (`acomquest/Saamayik`) via Hugging Face. This dataset provides ~53,000 parallel English-Sanskrit sentences, focusing on contemporary prose and pedagogical content, making it superior for modern translation tasks compared to classical poetry-focused corpora.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face `datasets` & `tokenizers`
- `tqdm`, `torchmetrics`, `tensorboard`

### Quick Start
```bash
# Clone the repository
git clone https://github.com/suhritghimire/Transformer-from-Scratch-Using-only-PyTorch
cd Transformer-from-Scratch-Using-only-PyTorch

# Install dependencies
pip install -r requirements.txt

# Start training
python train.py
```

---

## 📊 Monitoring Progress

Live training metrics are recorded using Tensorboard. To view the loss and validation metrics (BLEU, WER, CER) in real-time:

```bash
tensorboard --logdir runs/
```

---

## 👨‍💻 Author

**Suhrit Ghimire**  
*AI Enthusiast & Machine Learning Engineer*

This implementation reflects my passion for understanding the inner workings of state-of-the-art NLP models. Every line of code was crafted to ensure clarity, performance, and mathematical correctness.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
