# 🧠 BERT-Based Sentiment Classifier (Custom Implementation)

This repository contains a minimal yet effective custom implementation of a BERT-style sentiment classifier built from scratch using PyTorch. It was developed as part of the **Mental Satahi** Final Year Project, aiming to detect emotional sentiment from user text entries and track it over time.

## 🚀 Overview

- Implements a **Transformer Encoder** with 1 encoder block inspired by the "Attention is All You Need" paper.
- Achieved **80% validation accuracy** and **78 F1-score** on a public sentiment analysis dataset.
- Integrates with the larger Mental Satahi system for emotional tracking and reporting.

## 🧱 Model Architecture

- **Embedding Layer**: Word embeddings scaled by √(d_model)
- **Positional Encoding**: Sinusoidal, following original BERT/Transformer formulation
- **Multi-Head Self Attention**: 4 attention heads with residual connection
- **Feed Forward Network**: 2-layer MLP with ReLU and dropout
- **Layer Normalization** and **Residual Connections** in each sub-layer
- **Output Projection**: Mean pooling followed by a linear classifier to predict 5 sentiment classes

## ⚙️ Configuration

The model's core hyperparameters can be found in `configure.py`:
```python
{
  'd_model': 512,
  'h': 4,
  'seq_length': 100,
  'batch_size': 4,
  'd_ff': 768,
  'labels': 5,
  'number_of_blocks': 1
}
```
## 🧪 Dataset

- Used a public sentiment dataset from HuggingFace Hub.
- Dataset was preprocessed using a prebuilt tokenizer.

## 🧠 Training Summary
- Loss Function: CrossEntropyLoss

- Optimizer: Adam

- Batch Size: 4

- Max Sequence Length: 100

- Model Depth: 1 encoder block

- Training Epochs: 20

##  Input & Output Format
- Input: Tokenized sequences of text (token IDs).

- Masking: Supports padding mask input to self-attention.

- Output: Logits for 5 sentiment classes (can be mapped to labels like positive, neutral, negative, etc.)

## 📄 License
This project is part of an academic submission and is released under the MIT License for learning and non-commercial purposes.
