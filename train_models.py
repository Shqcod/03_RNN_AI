import pickle
from preprocess import load_and_preprocess
from model.model_rnn import build_rnn
from model.model_lstm import build_lstm

X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess()

# ----- Train RNN -----
rnn = build_rnn()
history_rnn = rnn.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128
)

rnn.save("model_h5/model_rnn.h5")
with open("history_model/history_rnn.pkl", "wb") as f:
    pickle.dump(history_rnn.history, f)

# ----- Train LSTM -----
lstm = build_lstm()
history_lstm = lstm.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128
)

lstm.save("model_h5/model_lstm.h5")
with open("history_model/history_lstm.pkl", "wb") as f:
    pickle.dump(history_lstm.history, f)

print("Training complete. Models saved!")
