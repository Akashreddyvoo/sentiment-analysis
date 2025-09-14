import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# === Load dataset ===
df = pd.read_csv("IMDB Dataset.csv")
print("✅ Data loaded successfully!")

# === Preprocess labels ===
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# === Tokenize text ===
vocab_size = 10000
max_len = 200
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])

sequences = tokenizer.texts_to_sequences(df['review'])
padded = pad_sequences(sequences, maxlen=max_len, truncating='post')

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(padded, df['sentiment'].values, test_size=0.2, random_state=42)

# === Build LSTM model ===
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# === Train model ===
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=128)

# === Save model ===
model.save("lstm_sentiment_model.h5")
print("✅ Model saved successfully!")

# === Plot accuracy ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# === Load model and make prediction on new review ===
loaded_model = load_model("lstm_sentiment_model.h5")
sample_review = ["This movie was a masterpiece and very emotional."]
sample_seq = tokenizer.texts_to_sequences(sample_review)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')
prediction = loaded_model.predict(sample_pad)

print("\n✅ Sentiment Prediction:", "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞")


# Load saved model
loaded_model = tf.keras.models.load_model("lstm_sentiment_model.h5")

# Test on a custom review
sample_review = ["This movie was absolutely wonderful!"]
sample_seq = tokenizer.texts_to_sequences(sample_review)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, truncating='post')

# Predict
prediction = loaded_model.predict(sample_pad)[0][0]
sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
print(f"\nReview: {sample_review[0]}")
print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.2f})")


# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {test_acc:.2%}, Loss: {test_loss:.4f}")


import pickle

# Save tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


# Load the saved model
loaded_model = tf.keras.models.load_model("lstm_sentiment_model.h5")

# Function to predict sentiment
def predict_sentiment(review_text):
    sequence = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(sequence, maxlen=max_len, truncating='post')
    prediction = loaded_model.predict(padded)[0][0]
    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    print(f"\nReview: {review_text}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.2f})")

# Try some custom reviews
predict_sentiment("This movie was amazing and beautifully shot!")
predict_sentiment("It was a waste of time. Poor acting and bad script.")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

import pickle

# Save tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Tokenizer saved as 'tokenizer.pickle'")

# Interactive mode for custom input
while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    predict_sentiment(user_input)
