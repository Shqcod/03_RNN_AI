import streamlit as st
import numpy as np
import random
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess


# ================================
# 🔧 Helper Functions
# ================================
def load_history(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def decode_review(encoded_review, index_to_word):
    return " ".join([index_to_word.get(i, "?") for i in encoded_review if i != 0])


# ================================
# 🚀 Streamlit UI
# ================================
st.set_page_config(page_title="RNN vs LSTM IMDB", layout="wide")

st.title("🎬 Perbandingan Model RNN vs LSTM pada Dataset IMDB")
st.write("Visualisasi evaluasi model berbasis dataset IMDB (tanpa input manual).")


# ================================
# 📌 Load dataset & preprocessing
# ================================
st.header("📌 Memuat Dataset")
with st.spinner("Sedang memproses dataset..."):
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess()

index_to_word = {v: k for k, v in tokenizer.word_index.items()}

st.success("Dataset berhasil dimuat!")


# ================================
# 📌 Load Model
# ================================
st.header("📌 Memuat Model yang Sudah Dilatih")

with st.spinner("Memuat model LSTM & RNN..."):
    model_lstm = load_model("model_h5/model_lstm.h5")
    model_rnn = load_model("model_h5/model_rnn.h5")

    history_lstm = load_history("history_model/history_lstm.pkl")
    history_rnn = load_history("history_model/history_rnn.pkl")

st.success("Model & history training berhasil dimuat!")


# ================================
# 📌 Prediksi
# ================================
st.header("📌 Melakukan Prediksi pada Test Set")

y_pred_lstm = (model_lstm.predict(X_test) > 0.5).astype("int32")
y_pred_rnn = (model_rnn.predict(X_test) > 0.5).astype("int32")

acc_lstm = accuracy_score(y_test, y_pred_lstm)
acc_rnn = accuracy_score(y_test, y_pred_rnn)

col1, col2 = st.columns(2)
col1.metric("Akurasi LSTM", f"{acc_lstm:.4f}")
col2.metric("Akurasi RNN", f"{acc_rnn:.4f}")


# ================================
# 📈 Training Curves
# ================================
st.header("📈 Training Curves")

def plot_training(history, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy curve
    ax[0].plot(history["accuracy"], label="Train Acc")
    ax[0].plot(history["val_accuracy"], label="Val Acc")
    ax[0].set_title(f"{title} Accuracy")
    ax[0].legend()

    # Loss curve
    ax[1].plot(history["loss"], label="Train Loss")
    ax[1].plot(history["val_loss"], label="Val Loss")
    ax[1].set_title(f"{title} Loss")
    ax[1].legend()

    st.pyplot(fig)


st.subheader("📌 LSTM Training Curve")
plot_training(history_lstm, "LSTM")

st.subheader("📌 RNN Training Curve")
plot_training(history_rnn, "RNN")


# ================================
# 🔢 Confusion Matrix
# ================================
st.header("📌 Confusion Matrix")

def plot_cm(cm, title):
    fig = plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)


cm_lstm = confusion_matrix(y_test, y_pred_lstm)
cm_rnn = confusion_matrix(y_test, y_pred_rnn)

col1, col2 = st.columns(2)
with col1:
    st.subheader("LSTM")
    plot_cm(cm_lstm, "LSTM Confusion Matrix")
with col2:
    st.subheader("RNN")
    plot_cm(cm_rnn, "RNN Confusion Matrix")


# ================================
# 📄 Classification Report
# ================================
st.header("📌 Classification Report")

col1, col2 = st.columns(2)
with col1:
    st.subheader("LSTM Report")
    st.text(classification_report(y_test, y_pred_lstm))

with col2:
    st.subheader("RNN Report")
    st.text(classification_report(y_test, y_pred_rnn))


# ================================
# 🔍 Contoh Prediksi
# ================================
st.header("🔍 Contoh Prediksi dari Dataset")

for i in random.sample(range(len(X_test)), 5):
    st.write("---")
    st.subheader(f"Sample Index: {i}")

    st.write("### 📝 Review:")
    st.write(decode_review(X_test[i], index_to_word))

    col1, col2, col3 = st.columns(3)
    col1.write(f"**Label Asli:** {'Positive' if y_test[i] == 1 else 'Negative'}")
    col2.write(f"**Prediksi LSTM:** {'Positive' if y_pred_lstm[i] == 1 else 'Negative'}")
    col3.write(f"**Prediksi RNN:** {'Positive' if y_pred_rnn[i] == 1 else 'Negative'}")


st.success("🎉 Semua visualisasi berhasil ditampilkan!")