"""
Diabetes Insight Miner - Streamlit Web Application
Aplikasi web interaktif untuk mengklasifikasikan teks PROs diabetes.
"""

import streamlit as st
import pandas as pd
import joblib
import re
import spacy
import os

# --- Konfigurasi ---
MODEL_PATH = os.path.join('models', 'diabetes_classifier.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# --- Fungsi Caching untuk Model ---
@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model dan vectorizer yang sudah dilatih."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        nlp = spacy.load("en_core_web_sm")
        return model, vectorizer, nlp
    except FileNotFoundError:
        st.error(f"Error: File model atau vectorizer tidak ditemukan. Pastikan '{MODEL_PATH}' dan '{VECTORIZER_PATH}' ada.")
        st.info("Jalankan skrip 'train_model.py' terlebih dahulu untuk membuat file model.")
        return None, None, None

# --- Fungsi Preprocessing ---
def preprocess_text(text, nlp):
    """
    Membersihkan dan memproses teks input menggunakan spaCy.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    doc = nlp(text)
    tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop and not token.is_punct and len(token.lemma_.strip()) > 2
    ]
    return " ".join(tokens)

# --- Tampilan Aplikasi ---
st.set_page_config(page_title="Diabetes Insight Miner", page_icon="🩺", layout="wide")

# Header
st.title("🩺 Diabetes Insight Miner")
st.markdown("Alat ini menggunakan Natural Language Processing (NLP) untuk mengklasifikasikan postingan forum online dari pasien diabetes ke dalam kategori yang relevan.")
st.markdown("---")

# Muat model
model, vectorizer, nlp = load_model_and_vectorizer()

if model and vectorizer and nlp:
    # Layout dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Masukkan Teks di Sini")
        user_input = st.text_area("Tempelkan judul dan isi postingan dari forum atau media sosial:", height=250, placeholder="Contoh: Saya baru saja memulai pengobatan Metformin dan merasa sangat mual. Apakah ini normal?")

        if st.button("🔬 Klasifikasikan Teks"):
            if user_input.strip():
                # Preprocess input
                processed_input = preprocess_text(user_input, nlp)
                
                # Vectorize input
                input_vector = vectorizer.transform([processed_input])
                
                # Prediksi
                prediction = model.predict(input_vector)
                prediction_proba = model.predict_proba(input_vector)
                
                # Tampilkan hasil di kolom kedua
                with col2:
                    st.subheader("✅ Hasil Klasifikasi")
                    st.success(f"**Kategori yang Diprediksi:** `{prediction[0]}`")
                    
                    st.subheader("Distribusi Probabilitas")
                    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probabilitas"])
                    st.dataframe(proba_df.T.style.format("{:.2%}").background_gradient(cmap='Greens'))
                    
                    st.info("**Catatan:** Model ini adalah Proof-of-Concept (PoC). Akurasi akan meningkat dengan lebih banyak data pelatihan.")

            else:
                st.warning("Harap masukkan teks untuk diklasifikasikan.")
else:
    st.warning("Aplikasi tidak dapat berjalan karena model tidak berhasil dimuat.")

# Footer
st.markdown("---")
st.markdown("Dibuat sebagai bagian dari proyek PoC untuk analisis data kesehatan.") 