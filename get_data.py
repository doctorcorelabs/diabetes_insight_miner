"""
Diabetes Insight Miner - Data Collection Script
Mengambil data dari subreddit r/diabetes menggunakan Reddit API
"""

import praw
import pandas as pd
import time
from datetime import datetime
import os
from reddit_config import CLIENT_ID, CLIENT_SECRET, USER_AGENT, SUBREDDIT_NAME, MAX_POSTS, TIME_FILTER

def setup_reddit_client():
    """Setup Reddit client dengan kredensial"""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        print(f"✅ Berhasil terhubung ke Reddit API")
        return reddit
    except Exception as e:
        print(f"❌ Error saat terhubung ke Reddit API: {e}")
        return None

def collect_posts(reddit, subreddit_name, max_posts=1000, time_filter='month'):
    """
    Mengumpulkan postingan dari subreddit
    
    Args:
        reddit: Reddit client instance
        subreddit_name: Nama subreddit
        max_posts: Jumlah maksimal postingan yang diambil
        time_filter: Filter waktu (hour, day, week, month, year, all)
    
    Returns:
        List of dictionaries berisi data postingan
    """
    posts_data = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"📊 Mengambil {max_posts} postingan dari r/{subreddit_name}")
        
        # Mengambil postingan berdasarkan filter waktu
        if time_filter == 'all':
            posts = subreddit.hot(limit=max_posts)
        else:
            posts = subreddit.top(time_filter=time_filter, limit=max_posts)
        
        count = 0
        for post in posts:
            try:
                # Ekstrak data postingan
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'body': post.selftext,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'author': str(post.author) if post.author else '[deleted]',
                    'url': post.url,
                    'permalink': post.permalink,
                    'is_self': post.is_self,
                    'over_18': post.over_18,
                    'spoiler': post.spoiler,
                    'stickied': post.stickied,
                    'subreddit': post.subreddit.display_name
                }
                
                posts_data.append(post_data)
                count += 1
                
                if count % 100 == 0:
                    print(f"📥 Telah mengambil {count} postingan...")
                
                # Rate limiting untuk menghindari API limit
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error saat mengambil postingan {post.id}: {e}")
                continue
        
        print(f"✅ Berhasil mengambil {len(posts_data)} postingan")
        
    except Exception as e:
        print(f"❌ Error saat mengumpulkan postingan: {e}")
    
    return posts_data

def save_to_csv(posts_data, filename='data/reddit_posts.csv'):
    """
    Menyimpan data postingan ke file CSV
    
    Args:
        posts_data: List of dictionaries berisi data postingan
        filename: Nama file output
    """
    try:
        # Buat direktori data jika belum ada
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert ke DataFrame
        df = pd.DataFrame(posts_data)
        
        # Simpan ke CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"💾 Data berhasil disimpan ke {filename}")
        print(f"📊 Total postingan: {len(df)}")
        print(f"📋 Kolom: {list(df.columns)}")
        
        # Tampilkan statistik dasar
        print("\n📈 Statistik Data:")
        print(f"   - Rata-rata skor: {df['score'].mean():.2f}")
        print(f"   - Rata-rata komentar: {df['num_comments'].mean():.2f}")
        print(f"   - Postingan dengan body: {df['body'].notna().sum()}")
        print(f"   - Postingan tanpa body: {df['body'].isna().sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error saat menyimpan data: {e}")
        return None

def main():
    """Fungsi utama"""
    print("🚀 Memulai pengambilan data dari Reddit...")
    print("=" * 50)
    
    # Setup Reddit client
    reddit = setup_reddit_client()
    if not reddit:
        print("❌ Gagal setup Reddit client. Pastikan kredensial API sudah benar.")
        return
    
    # Cek apakah kredensial masih default
    if CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        print("⚠️  PERINGATAN: Kredensial Reddit API masih default!")
        print("   Silakan edit file 'reddit_config.py' dan masukkan kredensial Anda.")
        print("   Kunjungi: https://www.reddit.com/prefs/apps")
        return
    
    # Ambil data postingan
    posts_data = collect_posts(
        reddit=reddit,
        subreddit_name=SUBREDDIT_NAME,
        max_posts=MAX_POSTS,
        time_filter=TIME_FILTER
    )
    
    if not posts_data:
        print("❌ Tidak ada data yang berhasil diambil.")
        return
    
    # Simpan data ke CSV
    df = save_to_csv(posts_data)
    
    if df is not None:
        print("\n✅ Pengambilan data selesai!")
        print("📁 File tersimpan di: data/reddit_posts.csv")
        print("🔄 Langkah selanjutnya: Pelabelan manual data")

if __name__ == "__main__":
    main() 