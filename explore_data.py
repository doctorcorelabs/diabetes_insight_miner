"""
Diabetes Insight Miner - Data Exploration Script
Menganalisis dan mengeksplorasi data yang telah dikumpulkan dari Reddit
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
import os

def load_data(filename='data/reddit_posts.csv'):
    """Memuat data dari file CSV"""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Data berhasil dimuat dari {filename}")
        print(f"📊 Total baris: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"❌ File {filename} tidak ditemukan")
        print("   Jalankan get_data.py terlebih dahulu untuk mengumpulkan data")
        return None
    except Exception as e:
        print(f"❌ Error saat memuat data: {e}")
        return None

def basic_statistics(df):
    """Menampilkan statistik dasar data"""
    print("\n" + "="*50)
    print("📈 STATISTIK DASAR DATA")
    print("="*50)
    
    print(f"📊 Total postingan: {len(df)}")
    print(f"📅 Rentang waktu: {df['created_utc'].min()} hingga {df['created_utc'].max()}")
    print(f"👥 Total author unik: {df['author'].nunique()}")
    print(f"🏆 Rata-rata skor: {df['score'].mean():.2f}")
    print(f"💬 Rata-rata komentar: {df['num_comments'].mean():.2f}")
    print(f"📝 Postingan dengan body: {df['body'].notna().sum()} ({df['body'].notna().sum()/len(df)*100:.1f}%)")
    print(f"🔗 Postingan link: {df['is_self'].sum()} ({df['is_self'].sum()/len(df)*100:.1f}%)")

def analyze_text_content(df):
    """Menganalisis konten teks"""
    print("\n" + "="*50)
    print("📝 ANALISIS KONTEN TEKS")
    print("="*50)
    
    # Analisis judul
    df['title_length'] = df['title'].str.len()
    df['title_words'] = df['title'].str.split().str.len()
    
    print(f"📏 Panjang judul rata-rata: {df['title_length'].mean():.1f} karakter")
    print(f"📝 Kata dalam judul rata-rata: {df['title_words'].mean():.1f} kata")
    
    # Analisis body (jika ada)
    if df['body'].notna().sum() > 0:
        df['body_length'] = df['body'].str.len()
        df['body_words'] = df['body'].str.split().str.len()
        
        body_stats = df[df['body'].notna()]
        print(f"📏 Panjang body rata-rata: {body_stats['body_length'].mean():.1f} karakter")
        print(f"📝 Kata dalam body rata-rata: {body_stats['body_words'].mean():.1f} kata")
    
    # Top 10 kata dalam judul
    print("\n🔝 TOP 10 KATA DALAM JUDUL:")
    all_titles = ' '.join(df['title'].dropna().astype(str)).lower()
    words = re.findall(r'\b\w+\b', all_titles)
    word_counts = pd.Series(words).value_counts().head(10)
    
    for i, (word, count) in enumerate(word_counts.items(), 1):
        print(f"   {i:2d}. {word:15s} ({count:3d} kali)")

def create_visualizations(df):
    """Membuat visualisasi data"""
    print("\n" + "="*50)
    print("📊 MEMBUAT VISUALISASI")
    print("="*50)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Buat direktori untuk gambar jika belum ada
    os.makedirs('data/plots', exist_ok=True)
    
    # 1. Distribusi skor
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribusi Skor Postingan')
    plt.xlabel('Skor')
    plt.ylabel('Frekuensi')
    plt.yscale('log')
    
    # 2. Distribusi jumlah komentar
    plt.subplot(2, 2, 2)
    plt.hist(df['num_comments'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribusi Jumlah Komentar')
    plt.xlabel('Jumlah Komentar')
    plt.ylabel('Frekuensi')
    plt.yscale('log')
    
    # 3. Scatter plot skor vs komentar
    plt.subplot(2, 2, 3)
    plt.scatter(df['score'], df['num_comments'], alpha=0.6, color='orange')
    plt.title('Skor vs Jumlah Komentar')
    plt.xlabel('Skor')
    plt.ylabel('Jumlah Komentar')
    plt.xscale('log')
    plt.yscale('log')
    
    # 4. Panjang judul vs skor
    plt.subplot(2, 2, 4)
    plt.scatter(df['title_length'], df['score'], alpha=0.6, color='purple')
    plt.title('Panjang Judul vs Skor')
    plt.xlabel('Panjang Judul (karakter)')
    plt.ylabel('Skor')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('data/plots/data_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Visualisasi disimpan di: data/plots/data_analysis.png")
    plt.show()

def show_sample_posts(df, n=5):
    """Menampilkan contoh postingan"""
    print("\n" + "="*50)
    print(f"📋 CONTOH {n} POSTINGAN")
    print("="*50)
    
    # Ambil postingan dengan skor tertinggi
    top_posts = df.nlargest(n, 'score')
    
    for i, (_, post) in enumerate(top_posts.iterrows(), 1):
        print(f"\n{i}. JUDUL: {post['title']}")
        print(f"   SKOR: {post['score']} | KOMENTAR: {post['num_comments']}")
        print(f"   AUTHOR: {post['author']}")
        print(f"   TANGGAL: {post['created_utc']}")
        
        if pd.notna(post['body']) and len(str(post['body'])) > 0:
            body_preview = str(post['body'])[:200] + "..." if len(str(post['body'])) > 200 else str(post['body'])
            print(f"   BODY: {body_preview}")
        print("-" * 80)

def suggest_categories(df):
    """Memberikan saran kategori berdasarkan analisis konten"""
    print("\n" + "="*50)
    print("🏷️  SARAN KATEGORI BERDASARKAN KONTEN")
    print("="*50)
    
    # Kata kunci untuk setiap kategori
    keywords = {
        'Efek Samping Obat': ['side effect', 'medication', 'drug', 'metformin', 'insulin', 'dose', 'prescription'],
        'Kesehatan Mental': ['depression', 'anxiety', 'stress', 'mental', 'therapy', 'counseling', 'support'],
        'Manajemen Diet': ['diet', 'food', 'carb', 'sugar', 'meal', 'nutrition', 'eating', 'keto'],
        'Dukungan & Motivasi': ['motivation', 'support', 'encouragement', 'success', 'progress', 'hope'],
        'Teknologi & Monitoring': ['glucose', 'monitor', 'device', 'app', 'technology', 'sensor', 'pump']
    }
    
    # Analisis judul dan body
    all_text = ' '.join(df['title'].dropna().astype(str)).lower()
    if df['body'].notna().sum() > 0:
        all_text += ' ' + ' '.join(df['body'].dropna().astype(str)).lower()
    
    print("📊 DISTRIBUSI KATEGORI BERDASARKAN KATA KUNCI:")
    for category, words in keywords.items():
        count = sum(all_text.count(word) for word in words)
        print(f"   {category}: {count} kemunculan")
    
    print("\n💡 SARAN UNTUK PELABELAN:")
    print("   1. Mulai dengan 200-300 postingan untuk pelabelan manual")
    print("   2. Fokus pada postingan dengan body yang lengkap")
    print("   3. Prioritaskan postingan dengan skor tinggi")
    print("   4. Gunakan kata kunci di atas sebagai panduan")

def main():
    """Fungsi utama"""
    print("🔍 MEMULAI EKSPLORASI DATA")
    print("="*50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Analisis dasar
    basic_statistics(df)
    
    # Analisis konten teks
    analyze_text_content(df)
    
    # Tampilkan contoh postingan
    show_sample_posts(df)
    
    # Saran kategori
    suggest_categories(df)
    
    # Buat visualisasi
    try:
        create_visualizations(df)
    except Exception as e:
        print(f"⚠️ Error saat membuat visualisasi: {e}")
    
    print("\n✅ Eksplorasi data selesai!")
    print("🔄 Langkah selanjutnya: Pelabelan manual data")

if __name__ == "__main__":
    main() 