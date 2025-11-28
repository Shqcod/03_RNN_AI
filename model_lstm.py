from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import load_and_preprocess

def build_lstm(max_words, max_len):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm():
    X_train, X_test, y_train, y_test, tokenizer, max_len, max_words = load_and_preprocess()

    model = build_lstm(max_words, max_len)

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

    print("[INFO] Training LSTM model...")
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] LSTM Test Accuracy = {acc:.4f}")

    print("[INFO] Saving model...")
    model.save("model_lstm.h5")

    print("[INFO] Model saved successfully.")
    return model


if __name__ == "__main__":
    train_lstm()
