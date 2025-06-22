"""
Diabetes Insight Miner - Model Training Script (using spaCy)
Melatih model klasifikasi teks untuk mengkategorikan postingan Reddit.
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Konfigurasi ---
LABELED_DATA_PATH = 'data/reddit_posts_labeled.csv'
MODEL_OUTPUT_DIR = 'models'
MODEL_FILENAME = 'diabetes_classifier.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Muat model spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' tidak ditemukan. Mengunduh...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """
    Membersihkan dan memproses teks input menggunakan spaCy.
    - Menghapus URL
    - Lemmatisasi
    - Menghapus stopwords dan tanda baca
    """
    if not isinstance(text, str):
        return ""
    
    # Hapus URL terlebih dahulu
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Proses teks dengan spaCy
    doc = nlp(text)
    
    # Lemmatisasi dan hapus stopwords/tanda baca
    tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop and not token.is_punct and len(token.lemma_.strip()) > 2
    ]
    
    return " ".join(tokens)

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    """Membuat dan menyimpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 Confusion matrix disimpan di: {filename}")

def main():
    """Fungsi utama untuk melatih dan mengevaluasi model."""
    print("🚀 Memulai proses pelatihan model dengan spaCy...")
    print("="*50)

    try:
        df = pd.read_csv(LABELED_DATA_PATH)
        print(f"✅ Berhasil memuat {len(df)} data berlabel dari {LABELED_DATA_PATH}")
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {LABELED_DATA_PATH}")
        return

    df.dropna(subset=['body', 'category'], inplace=True)
    df['text_to_process'] = df['title'] + ' ' + df['body']
    print(f"Jumlah data setelah membersihkan nilai kosong: {len(df)}")

    print("\n🔄 Memproses teks dengan spaCy...")
    df['processed_text'] = df['text_to_process'].apply(preprocess_text)
    print("✅ Teks selesai diproses.")

    X = df['processed_text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nMembagi data menjadi {len(X_train)} data latih dan {len(X_test)} data uji.")

    print("🔄 Membuat fitur teks dengan TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("✅ Fitur teks berhasil dibuat.")

    print("🤖 Melatih model Logistic Regression...")
    model = LogisticRegression(random_state=RANDOM_STATE, multi_class='ovr', solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    print("✅ Model berhasil dilatih.")

    print("\n" + "="*50)
    print("📈 EVALUASI MODEL")
    print("="*50)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi Model: {accuracy:.2%}\n")
    
    unique_labels = sorted(y.unique())
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, labels=unique_labels, zero_division=0))

    os.makedirs('data/plots', exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, classes=unique_labels, filename='data/plots/confusion_matrix.png')

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    vectorizer_path = os.path.join(MODEL_OUTPUT_DIR, VECTORIZER_FILENAME)
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n💾 Model berhasil disimpan di: {model_path}")
    print(f"💾 Vectorizer berhasil disimpan di: {vectorizer_path}")
    
    print("\n✅ Proses pelatihan selesai!")
    print("🔄 Langkah selanjutnya: Membuat aplikasi demo dengan Streamlit.")

if __name__ == "__main__":
    main() 