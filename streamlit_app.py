import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess


# ---------------------------------
# Load History Function
# ---------------------------------
def load_history(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


st.title("🔎 Perbandingan Model RNN vs LSTM pada Dataset IMDB")
st.write("Visualisasi evaluasi model berbasis dataset IMDB (tanpa input manual).")

# --- Load dataset ---
st.subheader("📌 Memuat dataset dan preprocessing...")
X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess()
st.success("Dataset berhasil dimuat!")

# --- Load Models ---
st.subheader("📌 Memuat Model...")
model_lstm = load_model("model_lstm.h5")
model_rnn = load_model("model_rnn.h5")

history_lstm = load_history("history_lstm.pkl")
history_rnn = load_history("history_rnn.pkl")

st.success("Model & history training berhasil dimuat!")

# --- Predict ---
st.subheader("📌 Membuat Prediksi...")
y_pred_lstm = (model_lstm.predict(X_test) > 0.5).astype("int32")
y_pred_rnn = (model_rnn.predict(X_test) > 0.5).astype("int32")

# --- Accuracy ---
acc_lstm = accuracy_score(y_test, y_pred_lstm)
acc_rnn = accuracy_score(y_test, y_pred_rnn)

st.header("📊 Perbandingan Akurasi Model")
st.write(f"**Akurasi LSTM:** {acc_lstm:.4f}")
st.write(f"**Akurasi RNN:** {acc_rnn:.4f}")

# ---------------------------------
# TRAINING PLOTS
# ---------------------------------
st.header("📈 Training Curves")

def plot_training(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax[0].plot(history['accuracy'], label="Train Acc")
    ax[0].plot(history['val_accuracy'], label="Val Acc")
    ax[0].set_title(f"{title} Accuracy")
    ax[0].legend()

    # Loss plot
    ax[1].plot(history['loss'], label="Train Loss")
    ax[1].plot(history['val_loss'], label="Val Loss")
    ax[1].set_title(f"{title} Loss")
    ax[1].legend()

    st.pyplot(fig)


st.subheader("📌 LSTM Training")
plot_training(history_lstm, "LSTM")

st.subheader("📌 RNN Training")
plot_training(history_rnn, "RNN")

# ---------------------------------
# Confusion Matrix
# ---------------------------------
st.header("📌 Confusion Matrix")

def plot_cm(cm, title):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

cm_lstm = confusion_matrix(y_test, y_pred_lstm)
cm_rnn = confusion_matrix(y_test, y_pred_rnn)

plot_cm(cm_lstm, "LSTM Confusion Matrix")
plot_cm(cm_rnn, "RNN Confusion Matrix")

# ---------------------------------
# Classification Reports
# ---------------------------------
st.header("📌 Classification Report")

st.subheader("LSTM Report")
st.text(classification_report(y_test, y_pred_lstm))

st.subheader("RNN Report")
st.text(classification_report(y_test, y_pred_rnn))

# ---------------------------------
# Contoh Prediksi Dataset
# ---------------------------------
st.header("🔍 Contoh Prediksi dari Dataset")

index_to_word = {v: k for k, v in tokenizer.word_index.items()}

def decode_review(encoded):
    return " ".join([index_to_word.get(i, "?") for i in encoded if i != 0])

st.subheader("Random Sample Predictions")

for i in random.sample(range(len(X_test)), 5):
    st.write("### Review:")
    st.write(decode_review(X_test[i]))

    st.write(f"**Label asli:** {'Positive' if y_test[i] == 1 else 'Negative'}")
    st.write(f"**Prediksi LSTM:** {'Positive' if y_pred_lstm[i] == 1 else 'Negative'}")
    st.write(f"**Prediksi RNN:** {'Positive' if y_pred_rnn[i] == 1 else 'Negative'}")
    st.write("---")

st.success("Selesai menampilkan evaluasi lengkap!")
