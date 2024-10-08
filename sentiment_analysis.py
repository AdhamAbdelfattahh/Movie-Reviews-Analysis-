import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained sentiment analysis model
model = load_model('sentiment_analysis_model.keras')

# Preprocessing parameters
max_words = 1000
max_len = 100

# Initialize the tokenizer (You should use the same tokenizer that was fitted during training)
tokenizer = Tokenizer(num_words=max_words)

# Create the GUI window
class SentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Movie Review Sentiment Analysis")
        
        self.label = tk.Label(master, text="Enter your movie review:")
        self.label.pack(pady=10)

        self.review_text = tk.Text(master, height=10, width=50)
        self.review_text.pack(pady=10)

        self.analyze_button = tk.Button(master, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def analyze_sentiment(self):
        review = self.review_text.get("1.0", tk.END).strip()
        if not review:
            messagebox.showwarning("Input Error", "Please enter a review.")
            return

        # Prepare the input for the model
        sequences = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequences, maxlen=max_len)

        # Predict sentiment
        prediction = model.predict(padded)
        sentiment_score = prediction[0][0]

        # Determine the sentiment and rating
        if sentiment_score >= 0.5:
            sentiment = "Positive"
            rating = round(sentiment_score * 5, 2)  # Scale to 5
        else:
            sentiment = "Negative"
            rating = round((1 - sentiment_score) * 5, 2)  # Scale to 5

        # Display the results
        self.result_label.config(text=f"Sentiment: {sentiment} (Rating: {rating}/5)")

# Create the main application window
root = tk.Tk()
app = SentimentAnalysisApp(root)
root.mainloop()
