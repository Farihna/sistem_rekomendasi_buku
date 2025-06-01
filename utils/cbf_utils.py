import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

def get_similar_books(query_title: str,
                      tfidf_matrix, 
                      all_titles_ordered: list, 
                      books_details_df: pd.DataFrame,
                      top_n=5):
    
    output_columns = ['Book Title', 'Book Author', 'image_url']

    if tfidf_matrix is None or not all_titles_ordered or books_details_df.empty:
        return pd.DataFrame(columns=output_columns)

    try:
        normalized_query_title = str(query_title).lower().strip()
        normalized_all_titles = [str(t).lower().strip() for t in all_titles_ordered]
        
        if normalized_query_title in normalized_all_titles:
            book_index = normalized_all_titles.index(normalized_query_title)
        else:
            if 'streamlit' in globals() or 'streamlit' in locals(): # Cek apakah st bisa diakses
                 st.warning(f"Judul buku '{query_title}' tidak ditemukan dalam daftar CBF.")
            else:
                 st.warning(f"Peringatan CBF: Judul buku '{query_title}' tidak ditemukan dalam daftar CBF.")
            return pd.DataFrame(columns=output_columns)
    except Exception as e:
        if 'streamlit' in globals() or 'streamlit' in locals():
            st.error(f"Error saat mencari indeks buku CBF: {e}")
        else:
            st.error(f"Error saat mencari indeks buku CBF: {e}")
        return pd.DataFrame(columns=output_columns)
      

    query_vector = tfidf_matrix[book_index]
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    num_to_fetch = top_n + 5
    similar_indices = cosine_similarities.argsort()[-(num_to_fetch):][::-1]
    
    similar_books_data = []
    count = 0
    for idx in similar_indices:
        if idx == book_index: 
            continue
        if count >= top_n: 
            break
        
        try:
            if idx < len(books_details_df) and idx < len(all_titles_ordered):
                title = all_titles_ordered[idx]
                book_info_row_series = books_details_df[books_details_df['title'] == title]

                if not book_info_row_series.empty:
                    book_info_row = book_info_row_series.iloc[0]
                    author = book_info_row.get('author', "Penulis tidak diketahui")
                    image_url_val = book_info_row.get('image_url', "")
                    if pd.isna(author): author = "Penulis tidak diketahui"
                    if pd.isna(image_url_val): image_url_val = ""
                else:
                    author = "Info detail tidak ditemukan"
                    image_url_val = ""

                similar_books_data.append({
                    'Book Title': title,
                    'Book Author': author,
                    'image_url': image_url_val
                })
                count += 1
        except Exception as e:
             if 'streamlit' in globals() or 'streamlit' in locals():
                st.warning(f"Error saat mengambil detail buku CBF untuk indeks {idx}: {e}")
             continue

    return pd.DataFrame(similar_books_data, columns=output_columns)