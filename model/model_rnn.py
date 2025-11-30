from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

def build_rnn(max_words=10000, max_len=200):
    model = Sequential([
        Embedding(max_words, 128),
        SimpleRNN(128, return_sequences=True, dropout=0.2),
        SimpleRNN(64, dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
