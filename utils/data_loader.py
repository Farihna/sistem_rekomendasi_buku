import streamlit as st
import pandas as pd
import pickle
import joblib 
import tensorflow as tf
from custom_models.recommender_net import RecommenderNet

def load_pickle_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Berkas tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat {file_path} dengan pickle: {e}")
    return None

def load_joblib_file(file_path):
    try:
        data = joblib.load(file_path)
        return data
    except FileNotFoundError:
        st.error(f"Berkas tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat {file_path} dengan joblib: {e}")
    return None

def load_csv_data(file_path, **kwargs):
    try:
        df = pd.read_csv(file_path, **kwargs)
        return df
    except FileNotFoundError:
        st.error(f"Berkas CSV tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat CSV {file_path}: {e}")
    return pd.DataFrame() # Kembalikan DataFrame kosong jika error

# @st.cache_resource
def load_recommender_model(num_users: int,
                           num_book: int,
                           weights_path: str,
                           embedding_size: int = 16,
                           dropout_rate: float = 0.2):

    try:
        # instance model baru dengan parameter yang sama
        loaded_model = RecommenderNet(num_users, num_book, embedding_size, dropout_rate=dropout_rate)
        dummy_batch_size = 1
        dummy_input = tf.zeros((dummy_batch_size, 2), dtype=tf.float32)

        # Panggil model dengan input dummy
        _ = loaded_model(dummy_input)

        print("Model arsitektur berhasil dibangun.")

        # Muat bobot ke dalam instance model yang baru dibuat
        loaded_model.load_weights(weights_path)
        print(f"Bobot model berhasil dimuat dari: {weights_path}")
        return loaded_model

    except FileNotFoundError:
        print(f"Error: File bobot tidak ditemukan di {weights_path}")
        return None
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None