# 🎬 IMDb Movie Review Sentiment Analysis (LSTM)

This project uses an LSTM (Long Short-Term Memory) neural network to perform **binary sentiment classification** (positive/negative) on the IMDb movie reviews dataset.

## 📌 Features

- Preprocessing with Keras `Tokenizer` and padding
- Binary classification using Bidirectional LSTM
- Interactive prediction via command-line input
- Model + tokenizer saving for future use

## 🧠 Model Summary

- Embedding → Bidirectional LSTM → Dense Layers
- Binary Crossentropy Loss + Adam Optimizer
- 3 training epochs, 200 sequence length

## 🛠️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt




