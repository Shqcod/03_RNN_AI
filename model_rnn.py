from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def build_rnn(max_words=10000, max_len=200):
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        SimpleRNN(64),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
