import streamlit as st
import pandas as pd

def tampilkan_rekomendasi_di_ui(df_rekomendasi: pd.DataFrame, judul_bagian: str):
    st.markdown(f"##### {judul_bagian}")

    if df_rekomendasi is None or df_rekomendasi.empty:
        st.info("Tidak ada buku yang dapat direkomendasikan saat ini untuk kriteria ini.")
        return

    if 'Book Title' not in df_rekomendasi.columns:
        st.error("Format data rekomendasi tidak sesuai: Kolom 'Book Title' tidak ditemukan.")
        st.caption("Menampilkan data mentah yang diterima:")
        st.dataframe(df_rekomendasi)
        return

    # Layout: 3 kolom per baris
    cols = st.columns(4)
    for i, (index, row) in enumerate(df_rekomendasi.iterrows()):
        col = cols[i % 4]  # Ambil kolom ke-0,1,2 secara bergantian

        with col:
            judul_buku = row['Book Title']
            penulis_buku = row.get('Book Author', 'Penulis tidak diketahui')
            if pd.isna(penulis_buku) or str(penulis_buku).strip() == "":
                penulis_buku = 'Penulis tidak diketahui'

            image_url = row.get('Image-URL-L', None)

            # Gambar buku
            if image_url and pd.notna(image_url) and isinstance(image_url, str) and image_url.strip() != "":
                try:
                    st.image(image_url, use_container_width=True)
                except Exception:
                    st.caption("Gambar tidak tersedia")
            else:
                st.caption("Gambar tidak tersedia")

            # Judul dan Penulis
            st.markdown(f"**{judul_buku}**")
            st.markdown(f"*{penulis_buku}*")

        # Ganti baris setelah 4 kolom
        if (i + 1) % 4 == 0 and i < len(df_rekomendasi) - 1:
            cols = st.columns(4)  # Reset ke baris baru
