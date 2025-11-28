import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 10000
max_len = 200

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(csv_path="IMDB Dataset.csv"):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(csv_path)

    print("[INFO] Cleaning text...")
    df["review_clean"] = df["review"].apply(clean_text)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X = df["review_clean"].values
    y = df["sentiment"].values

    print("[INFO] Tokenizing...")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)

    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding="post")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42
    )

    print("[INFO] Preprocessing complete.")
    return X_train, X_test, y_train, y_test, tokenizer
