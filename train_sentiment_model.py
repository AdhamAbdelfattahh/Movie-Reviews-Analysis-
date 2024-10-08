import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (for demonstration purposes)
reviews = [
    'I loved this movie!',
    'This was terrible',
    'Great film!',
    'I would not recommend it',
    'Absolutely fantastic experience',
    'Not worth the time',
    'A masterpiece!',
    'I hated this film'
]
sentiments = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=10)

# Build a simple model
model = models.Sequential([
    layers.Embedding(10000, 8, input_length=10),
    layers.GlobalAveragePooling1D(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(sentiments), epochs=10)

# Save the model
model.save('sentiment_analysis_model.keras')
