"""
Diabetes Insight Miner - Data Labeling Preparation Script
Mempersiapkan data untuk pelabelan manual
"""

import pandas as pd
import os

def prepare_data_for_labeling(
    input_filename='data/reddit_posts.csv', 
    output_filename='data/reddit_posts_to_label.csv', 
    num_samples=300
):
    """
    Memuat data, memilih sampel, dan menyiapkannya untuk pelabelan
    
    Args:
        input_filename: File CSV input
        output_filename: File CSV output
        num_samples: Jumlah sampel yang akan dipilih
    """
    try:
        # Load data
        df = pd.read_csv(input_filename)
        print(f"✅ Berhasil memuat {len(df)} postingan dari {input_filename}")
        
        # Filter postingan yang memiliki body
        df_with_body = df[df['body'].notna() & (df['body'].str.len() > 20)]
        print(f"📊 Menemukan {len(df_with_body)} postingan dengan body yang signifikan")
        
        # Ambil sampel acak
        if len(df_with_body) < num_samples:
            print(f"⚠️ Jumlah postingan yang valid ({len(df_with_body)}) kurang dari sampel yang diminta ({num_samples})")
            sample_df = df_with_body.copy()
        else:
            sample_df = df_with_body.sample(n=num_samples, random_state=42)
        
        print(f"📝 Mengambil {len(sample_df)} sampel untuk pelabelan")
        
        # Tambahkan kolom kategori kosong
        sample_df['category'] = ""
        
        # Pilih kolom yang relevan
        labeling_df = sample_df[['id', 'title', 'body', 'category']]
        
        # Simpan ke CSV
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        labeling_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"💾 Berhasil menyimpan data untuk pelabelan di {output_filename}")
        print("\n" + "="*50)
        print("TUGAS ANDA:")
        print("1. Buka file 'data/reddit_posts_to_label.csv' di Excel atau Google Sheets.")
        print("2. Isi kolom 'category' untuk setiap baris dengan salah satu kategori berikut:")
        print("   - Efek Samping Obat")
        print("   - Kesehatan Mental")
        print("   - Manajemen Diet")
        print("   - Dukungan & Motivasi")
        print("   - Teknologi & Monitoring")
        print("   - Lainnya")
        print("3. Setelah selesai, simpan file tersebut sebagai 'data/reddit_posts_labeled.csv'.")
        print("4. Beri tahu saya jika Anda sudah selesai, dan kita akan melanjutkan ke tahap training model.")
        print("="*50)

    except FileNotFoundError:
        print(f"❌ File {input_filename} tidak ditemukan")
        print("   Jalankan get_data.py terlebih dahulu")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    prepare_data_for_labeling() 