import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path file CSV
csv_file = "ikan_air_tawar.csv"

# Pastikan file CSV tersedia
if not os.path.exists(csv_file):
    messagebox.showerror("Error", f"File {csv_file} tidak ditemukan!")
    exit()

# Membaca data CSV ke Pandas DataFrame
data = pd.read_csv(csv_file)

def preprocess_data():
    """Gabungkan semua kolom menjadi satu teks untuk setiap ikan."""
    data['combined'] = data.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    return data

data = preprocess_data()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
corpus = data['combined'].tolist()
tfidf_matrix = vectorizer.fit_transform(corpus)

def find_similar_fish():
    """Cari ikan dengan teks input paling mirip menggunakan cosine similarity."""
    user_input = entry_input.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Peringatan", "Input tidak boleh kosong!")
        return

    # Transformasi input user ke dalam TF-IDF
    user_vector = vectorizer.transform([user_input])

    # Hitung kemiripan cosine
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Ambil 5 hasil teratas
    top_indices = cosine_sim.argsort()[-5:][::-1]
    top_scores = cosine_sim[top_indices]

    # Bersihkan hasil sebelumnya
    for row in tree.get_children():
        tree.delete(row)

    # Tampilkan hasil hanya jika skornya relevan
    for idx, score in zip(top_indices, top_scores):
        if score > 0.1:  # Hanya tampilkan skor relevan
            result = data.iloc[idx]
            tree.insert("", tk.END, values=(result['Ikan'], result['Habitat'], result['Ukuran'],
                                            result['Bentuk Tubuh'], result['Warna'],
                                            result['Nilai Ekonomis'], result['Tingkah Laku'],
                                            f"{score:.2f}"))

    if not any(score > 0.1 for score in top_scores):
        messagebox.showinfo("Hasil", "Tidak ada ikan yang relevan dengan input Anda.")

# GUI Utama
root = tk.Tk()
root.title("Chat Bot Pencarian Ikan Air Tawar")
root.geometry("900x500")

# Input Area
tk.Label(root, text="Masukkan Deskripsi atau Kata Kunci:").pack(pady=5)
entry_input = tk.Text(root, height=4, width=80)
entry_input.pack(pady=5)

# Tombol Cari
tk.Button(root, text="Cari Ikan", command=find_similar_fish, bg="blue", fg="white").pack(pady=5)

# Frame untuk TreeView dan Scrollbar
frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Scrollbar Vertikal
scroll_y = ttk.Scrollbar(frame, orient="vertical")
scroll_y.pack(side="right", fill="y")

# Scrollbar Horizontal
scroll_x = ttk.Scrollbar(frame, orient="horizontal")
scroll_x.pack(side="bottom", fill="x")

# Hasil Tabel (TreeView)
columns = ["Ikan", "Habitat", "Ukuran", "Bentuk Tubuh", "Warna", "Nilai Ekonomis", "Tingkah Laku", "Skor"]
tree = ttk.Treeview(frame, columns=columns, show='headings', yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

# Konfigurasi kolom
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=120)

tree.pack(fill=tk.BOTH, expand=True)

# Menghubungkan scrollbar dengan TreeView
scroll_y.config(command=tree.yview)
scroll_x.config(command=tree.xview)

# Menjalankan Aplikasi
root.mainloop()
