"""
Diabetes Insight Miner - Labeled Data Simulation Script
Membuat file data berlabel simulasi berdasarkan distribusi yang ditentukan pengguna.
"""

import pandas as pd
import numpy as np
import os

def simulate_labeled_data(
    input_filename='data/reddit_posts_to_label.csv',
    output_filename='data/reddit_posts_labeled.csv'
):
    """
    Membuat file CSV berlabel simulasi.
    
    Args:
        input_filename: File CSV dengan data yang akan diberi label.
        output_filename: Nama file output untuk data berlabel.
    """
    # Distribusi kategori yang ditentukan oleh pengguna
    category_distribution = {
        'Medication & Treatment': 115,
        'Devices & Technology': 102,
        'Lifestyle & Diet': 44,
        'General Discussion': 16,
        'Symptoms & Complications': 13,
        'Support & Experience': 7,
        'Diagnosis & Prevention': 3
    }
    
    total_labels = sum(category_distribution.values())
    print(f"Total label yang akan disimulasikan: {total_labels}")

    try:
        # Muat data yang telah disiapkan untuk pelabelan
        df = pd.read_csv(input_filename)
        
        # Pastikan jumlah data cukup
        if len(df) < total_labels:
            raise ValueError(f"Data input ({len(df)}) lebih sedikit dari total label ({total_labels})")

        # Ambil subset data sejumlah total label
        df_to_label = df.head(total_labels).copy()
        
        # Buat daftar label sesuai distribusi
        labels = []
        for category, count in category_distribution.items():
            labels.extend([category] * count)
        
        # Acak label agar tidak berurutan
        np.random.shuffle(labels)
        
        # Masukkan label ke dalam DataFrame
        df_to_label['category'] = labels
        
        # Simpan ke file CSV baru
        df_to_label.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"\n✅ Berhasil membuat file data berlabel simulasi di: {output_filename}")
        print("📊 Distribusi kategori dalam file simulasi:")
        print(df_to_label['category'].value_counts())
        
    except FileNotFoundError:
        print(f"❌ File {input_filename} tidak ditemukan.")
    except Exception as e:
        print(f"❌ Terjadi error: {e}")

if __name__ == "__main__":
    simulate_labeled_data() 