
import pandas as pd
import numpy as np

def siapkan_input_model_cf(user_id_input, 
                           df_ratings_global: pd.DataFrame,
                           df_books_global: pd.DataFrame, 
                           isbn_encoding_global: dict,  
                           user_encoding_global: dict): 
    
    book_readed_by_user = df_ratings_global[df_ratings_global['User-ID'] == user_id_input]

    isbns_not_read_series = df_books_global[~df_books_global['ISBN'].isin(book_readed_by_user['ISBN'].values)]['ISBN']
    
    candidate_isbns_str = list(
        set(isbns_not_read_series.unique())
        .intersection(set(isbn_encoding_global.keys()))
    )

    if not candidate_isbns_str:
        return None, []

    user_encoded_id = user_encoding_global.get(user_id_input)
    if user_encoded_id is None:
        return None, []

    book_encoded_ids_list_of_lists = []
    valid_candidate_isbns_for_array = []
    for isbn_str in candidate_isbns_str:
        encoded_id = isbn_encoding_global.get(isbn_str)
        if encoded_id is not None:
            book_encoded_ids_list_of_lists.append([encoded_id])
            valid_candidate_isbns_for_array.append(isbn_str)

    if not book_encoded_ids_list_of_lists:
        return None, []

    user_id_repeated_array = np.full((len(book_encoded_ids_list_of_lists), 1), user_encoded_id, dtype=np.int32)
    book_ids_array = np.array(book_encoded_ids_list_of_lists, dtype=np.int32)
    user_book_array_intermediate = np.hstack((user_id_repeated_array, book_ids_array))
    user_book_array_output = user_book_array_intermediate.astype(np.float32)
    
    return user_book_array_output, valid_candidate_isbns_for_array


def get_cf_recommendation(user_id, 
                          loaded_cf_model, 
                          df_ratings_data: pd.DataFrame, 
                          df_books_details: pd.DataFrame, 
                          isbn_encoder_cf: dict, 
                          user_encoder_cf: dict, 
                          top_n=10):

    output_columns = ['Book Title', 'Book Author', 'image_url']

    if loaded_cf_model is None:
        return pd.DataFrame(columns=output_columns)

    user_book_array_for_pred, candidate_isbns_for_pred = siapkan_input_model_cf(
        user_id,
        df_ratings_data,
        df_books_details, 
        isbn_encoder_cf,
        user_encoder_cf
    )

    if user_book_array_for_pred is None or not candidate_isbns_for_pred:
        return pd.DataFrame(columns=output_columns)

    model_input = user_book_array_for_pred 
    
    output_tensor = 'output_0'

    try:
        predictions_output = loaded_cf_model(model_input)
        if isinstance(predictions_output, dict):
            if output_tensor in predictions_output:
                predictions = predictions_output[output_tensor].numpy().flatten()
            else:
                output_keys = list(predictions_output.keys())
                if len(output_keys) == 1:
                    predictions = predictions_output[output_keys[0]].numpy().flatten()
                else:
                    raise ValueError(f"Output tensor CF '{output_tensor}' tidak ditemukan. Keys: {output_keys}")
        else:
            predictions = predictions_output.numpy().flatten()
            
    except Exception as e:
        return pd.DataFrame(columns=output_columns)

    top_indices = predictions.argsort()[::-1][:top_n]
    top_isbns_recommended = [candidate_isbns_for_pred[i] for i in top_indices]
    
    recommended_books_data = []
    for isbn_str in top_isbns_recommended:
        book_info = df_books_details[df_books_details['ISBN'] == isbn_str]
        
        if not book_info.empty:
            title = book_info['title'].iloc[0] 
            author = book_info.get('author', pd.Series(["Penulis tidak diketahui"])).iloc[0]
            image_url_val = book_info.get('image_url', pd.Series([""])).iloc[0] # Asumsi kolom 'image_url' sudah ada
            
            if pd.isna(author): author = "Penulis tidak diketahui"
            if pd.isna(image_url_val): image_url_val = ""
        else:
            title = f"Judul untuk ISBN {isbn_str} tidak ditemukan"
            author = "N/A"
            image_url_val = ""
            
        recommended_books_data.append({
            'Book Title': title,
            'Book Author': author,
            'image_url': image_url_val
        })
        
    return pd.DataFrame(recommended_books_data, columns=output_columns)