import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===========================
# Load Tokenizer
# ===========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # harus sama dengan preprocess.py

# ===========================
# Load Models
# ===========================
@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model("model_lstm.h5")

@st.cache_resource
def load_rnn():
    return tf.keras.models.load_model("model_rnn.h5")

model_lstm = load_lstm()
model_rnn = load_rnn()


# ===========================
# Function Predict
# ===========================
def predict_sentiment(text, model):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(pad)[0][0]
    return float(pred)


# ===========================
# UI Streamlit
# ===========================
st.set_page_config(page_title="Sentiment Analysis: LSTM vs RNN", layout="wide")

st.title("📘 Sentiment Analysis IMDB — LSTM vs RNN")
st.write("Bandingkan hasil prediksi sentimen antara model **LSTM** dan **RNN (SimpleRNN)**.")

st.markdown("---")

# ===========================
# Input Box
# ===========================
text_input = st.text_area(
    "Masukkan review film:",
    height=180,
    placeholder="Contoh: The movie was absolutely amazing!"
)

col1, col2 = st.columns(2)

if st.button("🔍 Analisis Sentimen"):
    if text_input.strip() == "":
        st.warning("Tolong masukkan teks review terlebih dahulu.")
    else:
        with st.spinner("Memproses..."):
            lstm_score = predict_sentiment(text_input, model_lstm)
            rnn_score = predict_sentiment(text_input, model_rnn)

        # LSTM result
        with col1:
            st.subheader("📘 Hasil Model LSTM")
            lstm_label = "Positive 😊" if lstm_score >= 0.5 else "Negative 😞"
            st.metric(label="Sentiment", value=lstm_label, delta=f"{lstm_score:.4f}")

        # RNN result
        with col2:
            st.subheader("📙 Hasil Model RNN (SimpleRNN)")
            rnn_label = "Positive 😊" if rnn_score >= 0.5 else "Negative 😞"
            st.metric(label="Sentiment", value=rnn_label, delta=f"{rnn_score:.4f}")

        st.markdown("---")

        st.subheader("📊 Perbandingan Skor")
        st.write(f"**LSTM Output:** {lstm_score:.4f}")
        st.write(f"**RNN Output:** {rnn_score:.4f}")

        st.info(
            "Catatan: Angka di atas menunjukkan probabilitas sentimen positif. "
            "Semakin dekat ke 1 berarti lebih positif."
        )

else:
    st.write("Masukkan teks review, lalu tekan tombol *Analisis Sentimen*.")
