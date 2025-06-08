import pandas as pd
import numpy as np

def get_cf_recommendations(user_id_input,
                           df_ratings_global: pd.DataFrame,
                           df_clean_books_global: pd.DataFrame,
                           isbn_encoding_global: dict,
                           user_encoding_global: dict,
                           model,
                           top_n):

    # Ambil buku yang sudah dibaca pengguna
    book_readed_by_user = df_ratings_global[df_ratings_global['User-ID'] == user_id_input]

    # Siapkan DataFrame buku yang sudah dibaca pengguna 
    df_book_readed_by_user_data = []
    if not book_readed_by_user.empty:
        top_book_user_isbns = (
            book_readed_by_user.sort_values(
                by='Book-Rating',
                ascending=False
            )
            .head(10)['ISBN'].values
        )

        for isbn_val in top_book_user_isbns:
            book_detail_series = df_clean_books_global[df_clean_books_global['ISBN'] == isbn_val]
            if not book_detail_series.empty:
                book_detail = book_detail_series.iloc[0]
                df_book_readed_by_user_data.append([book_detail['Book-Title'], book_detail['Book-Author'], book_detail['Image-URL-L']])

    output_columns_read = ['Book Title', 'Book Author', 'User Rating']
    df_book_readed_by_user = pd.DataFrame(df_book_readed_by_user_data, columns=output_columns_read)

    # Siapkan input untuk model prediksi
    if user_id_input not in user_encoding_global:
        print(f"Pengguna dengan ID {user_id_input} tidak ditemukan dalam pemetaan pengguna.")
        return df_book_readed_by_user, pd.DataFrame()

    user_encoded_id = user_encoding_global[user_id_input]

    # Dapatkan semua ISBN yang ada di clean_books DAN di isbn_encoding
    all_valid_isbns = list(
        set(df_clean_books_global['ISBN'].unique())
        .intersection(set(isbn_encoding_global.keys()))
    )

    # Filter ISBN yang sudah dibaca oleh pengguna
    book_readed_isbns = book_readed_by_user['ISBN'].unique()
    book_not_readed_original_isbns = [
        isbn for isbn in all_valid_isbns if isbn not in book_readed_isbns
    ]

    if not book_not_readed_original_isbns:
        print(f"Tidak ada buku yang belum dibaca oleh pengguna {user_id_input} yang dapat diprediksi.")
        return df_book_readed_by_user, pd.DataFrame()

    # Ubah ISBN asli buku yang belum dibaca menjadi ID ter-encode
    book_not_readed_encoded = [isbn_encoding_global.get(isbn) for isbn in book_not_readed_original_isbns]
    book_not_readed_encoded = [encoded_id for encoded_id in book_not_readed_encoded if encoded_id is not None]

    if not book_not_readed_encoded:
        print(f"Tidak ada ISBN yang belum dibaca oleh pengguna {user_id_input} yang ditemukan di encoding.")
        return df_book_readed_by_user, pd.DataFrame()

    # Membuat user_book_array (N, 2) untuk input model
    user_ids_array = np.full((len(book_not_readed_encoded), 1), user_encoded_id)
    books_ids_array = np.array(book_not_readed_encoded).reshape(-1, 1)
    user_book_array = np.hstack((user_ids_array, books_ids_array))

    # Prediksi Rating dan Pengambilan Top N
    user_book_array_input = user_book_array.astype(np.float32)
    predicted_ratings_normalized = model.predict(user_book_array_input).flatten()

    num_recommendations = min(top_n, len(predicted_ratings_normalized))
    top_ratings_indices = predicted_ratings_normalized.argsort()[-num_recommendations:][::-1]

    # Dapatkan ID buku ter-encode yang direkomendasikan
    recommended_book_ids_encoded = [book_not_readed_encoded[i] for i in top_ratings_indices]

    encoded_to_original_isbn_map = {v: k for k, v in isbn_encoding_global.items()}

    recommended_original_isbns = [encoded_to_original_isbn_map.get(encoded_id)
                                  for encoded_id in recommended_book_ids_encoded]

    recommended_book_data_final = []
    for i, original_isbn in enumerate(recommended_original_isbns):
        if original_isbn:
            book_detail_series = df_clean_books_global[df_clean_books_global['ISBN'] == original_isbn]
            if not book_detail_series.empty:
                book_detail = book_detail_series.iloc[0]
                recommended_book_data_final.append({
                    'Book Title': book_detail['Book-Title'],
                    'Book Author': book_detail['Book-Author'],
                    'Image-URL-L': book_detail['Image-URL-L']
                })
            else:
                recommended_book_data_final.append({
                    'Book Title': f"Detail tidak ditemukan untuk ISBN: {original_isbn}",
                    'Book Author': '-',
                    'Image-URL-L': '' 
                })
        else:
            recommended_book_data_final.append({
                'Book Title': f"ISBN asli tidak ditemukan untuk encoded_id: {recommended_book_ids_encoded[i]}",
                'Book Author': '-',
                'Image-URL-L': ''
            })

    # Membuat DataFrame untuk output rekomendasi
    output_columns_rec = ['Book Title', 'Book Author', 'Image-URL-L']
    df_recommended_books = pd.DataFrame(recommended_book_data_final, columns=output_columns_rec)
    
    return df_recommended_books