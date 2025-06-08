import streamlit as st

from utils.data_loader import (
    load_pickle_file,                # Untuk memuat user_encoding.pkl, isbn_encoding.pkl (CF), book_titles.pkl (jika dict)
    load_joblib_file,                # Untuk memuat cbf_tfidf_matrix.pkl jika disimpan dengan joblib
    load_csv_data,                   # Untuk memuat .csv
    load_recommender_model,          # Untuk memuat model CF sebagai weight
)
from utils.cf_utils import get_cf_recommendations
from utils.cbf_utils import get_similar_books
from utils.ui_components import tampilkan_rekomendasi_di_ui

# Streamlit UI
st.set_page_config(page_title="Rekomendasi Buku", layout="wide") 
st.title("Aplikasi Rekomendasi Buku")

# Aset CF
user_encoding = load_pickle_file("assets/encoders/user_encoding.pkl")
isbn_encoding = load_pickle_file("assets/encoders/isbn_encoding.pkl")
df_ratings = load_csv_data("assets/data/df_ratings.csv")
book_df = load_csv_data("assets/data/clean_books.csv")

# Model CF (weight)
num_users_book_count = load_pickle_file("assets/data/user_book_counts.pkl")
num_users_book_count = load_pickle_file("assets/data/user_book_counts.pkl")
# load jumlah user dan buku 
num_users = num_users_book_count.get('num_user', 0)
num_books = num_users_book_count.get('num_book', 0)
# load model
cf_model = load_recommender_model(num_users, num_books, "assets/cf_model/cf.weights.h5")

# Aset CBF
df_books_cbf = load_csv_data("assets/data/books_cbf.csv")
cbf_tfidf_matrix = load_joblib_file("assets/cbf_model/cbf_tfidf_matrix.pkl")
book_titles = load_pickle_file("assets/cbf_model/book_titles.pkl")

if df_books_cbf is not None and not df_books_cbf.empty:
    book_titles = df_books_cbf['Book-Title'].tolist()
    if cbf_tfidf_matrix is not None and book_titles is not None:
        if cbf_tfidf_matrix.shape[0] != len(book_titles):
            st.error(f"CRITICAL ERROR: Ketidakcocokan jumlah baris matriks TF-IDF CBF ({cbf_tfidf_matrix.shape[0]}) dan daftar judul CBF ({len(book_titles)}). Fitur CBF mungkin tidak berfungsi.")
            book_titles = None 
    elif book_titles is None:
         st.error("Gagal membuat book_titles karena DF_BOOKS_CBF kosong atau None.")
else:
    st.error("DataFrame buku untuk CBF (DF_BOOKS_CBF) gagal dimuat atau kosong.")


# Simpan status login di session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# user ID 
list_user_id = df_ratings['User-ID']

# Jika belum login, tampilkan form login
if not st.session_state.logged_in:
    st.subheader("🔐 Silakan Login Terlebih Dahulu")
    selected_user_id_cf = st.selectbox("Pilih User ID untuk Login:", options=list_user_id, index=0)

    if st.button("🔓 Login", key="login_button"):
        st.session_state.logged_in = True
        st.session_state.user_id = selected_user_id_cf
        st.rerun()

else:
    user_id = st.session_state.user_id

    st.success(f"Login sebagai User ID: {user_id}")
    st.markdown("---")
    st.subheader("🔍 Cari Buku Serupa")

    # Fitur pencarian CBF
    if all([df_books_cbf is not None, cbf_tfidf_matrix is not None, book_titles]):
        book_title_options = df_books_cbf['title'].drop_duplicates().tolist()
        selected_book_title = st.selectbox("Pilih Judul Buku:", options=book_title_options, key="cbf_selectbox")
        num_recs_cbf = st.slider("Jumlah Rekomendasi:", 1, 15, 5, key="cbf_slider")

        if st.button("Cari Buku", key="cbf_button"):
            if selected_book_title:
                similar_df = get_similar_books(
                    selected_book_title,
                    cbf_tfidf_matrix,
                    book_titles,
                    df_books_cbf,
                    top_n=num_recs_cbf,
                )
                tampilkan_rekomendasi_di_ui(similar_df, f"Buku serupa dengan '{selected_book_title}':")
    else:
        st.error("Aset untuk Content-Based Filtering tidak lengkap. Periksa pemuatan data.")

    # Cek apakah semua aset tersedia untuk CF
    if all([user_encoding, isbn_encoding, df_ratings is not None, cf_model, book_df is not None]):

        # Ambil dan tampilkan rekomendasi CF
        rekomendasi_df = get_cf_recommendations(
            user_id,
            df_ratings,
            book_df,
            isbn_encoding,
            user_encoding,
            cf_model,
            20
        )
        st.subheader("📌 Rekomendasi Berdasarkan Preferensi Anda")
        tampilkan_rekomendasi_di_ui(rekomendasi_df, f"Rekomendasi untuk User: {user_id}")
    else:
        st.error("Aset untuk Collaborative Filtering tidak lengkap. Periksa pemuatan data.")
