
# 🎬 IMDb LSTM Sentiment Analysis

This project uses an LSTM-based deep learning model to classify movie reviews from the [IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) as **Positive** or **Negative**.

---

## 🚀 Project Structure

```
sentiment-analysis/
│
├── imdb_lstm_sentiment.py         # Main script to train + run predictions
├── lstm_sentiment_model.h5        # Saved LSTM model
├── tokenizer.pickle               # Tokenizer used for preprocessing
├── IMDB Dataset.csv               # Training dataset (63MB)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Files to ignore
```

---

## 🧠 Model Architecture

- **Embedding Layer** with 64 dimensions
- **Bidirectional LSTM** with 64 units
- **Dense layer** with 64 neurons (ReLU)
- **Final Output**: 1 neuron (Sigmoid) for binary classification

---

## 📊 Dataset

- 50,000 labeled IMDb movie reviews
- Balanced dataset (50% positive, 50% negative)
- Source: [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🧪 How to Train

```bash
# 1. Ensure IMDB Dataset.csv is present in the same folder
# 2. Run training script
python imdb_lstm_sentiment.py
```

This will:
- Train the model
- Save the model as `lstm_sentiment_model.h5`
- Save the tokenizer as `tokenizer.pickle`

---

## ✨ How to Run Inference

```bash
python imdb_lstm_sentiment.py
# Then enter a review interactively:
> This movie was thrilling and emotional!
> Predicted Sentiment: Positive 😊 (Confidence: 0.89)
```

---

## 💾 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Notes

- `IMDB Dataset.csv` is >50MB. GitHub may warn on file size.
- You can automate download via Kaggle using `kagglehub` or skip pushing it to GitHub.

---

## 📅 Last Updated

September 14, 2025

---

## 📌 Author

Akash Reddy Vootkuri  
[GitHub Repo](https://github.com/Akashreddyvoo/sentiment-analysis)
