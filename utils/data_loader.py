import streamlit as st
import pandas as pd
import pickle
import joblib 
import tensorflow as tf
from tensorflow import keras 

def load_pickle_file(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Berhasil memuat: {file_path}")
        return data
    except FileNotFoundError:
        st.error(f"Berkas tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat {file_path} dengan pickle: {e}")
    return None

def load_joblib_file(file_path):
    try:
        data = joblib.load(file_path)
        print(f"Berhasil memuat: {file_path}")
        return data
    except FileNotFoundError:
        st.error(f"Berkas tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat {file_path} dengan joblib: {e}")
    return None

def load_csv_data(file_path, **kwargs):
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"Berhasil memuat: {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"Berkas CSV tidak ditemukan: {file_path}")
    except Exception as e:
        st.error(f"Error saat memuat CSV {file_path}: {e}")
    return pd.DataFrame() # Kembalikan DataFrame kosong jika error

def load_cf_keras_model(model_directory_path, custom_objects=None):
    print(f"Mencoba memuat Keras model dari: {model_directory_path}...")
    try:
        model = tf.keras.models.load_model(
            model_directory_path,
            custom_objects=custom_objects
        )
        print(f"Model Keras dari {model_directory_path} berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat Keras model dari {model_directory_path}. Error: {e}")
        print(f"Error detail saat memuat Keras model: {e}")
        import traceback
        print(traceback.format_exc())
    return None

def load_tfsmlayer_model(model_directory_path, call_endpoint='serving_default'):
    print(f"Mencoba memuat model sebagai TFSMLayer dari: {model_directory_path}...")
    try:
        model_layer = keras.layers.TFSMLayer(model_directory_path, call_endpoint=call_endpoint)
        print(f"TFSMLayer dari {model_directory_path} berhasil diinisialisasi.")
        return model_layer
    except Exception as e:
        st.error(f"Gagal menginisialisasi TFSMLayer dari {model_directory_path}. Error: {e}")
        print(f"Error detail saat init TFSMLayer: {e}")
        import traceback
        print(traceback.format_exc())
    return None

def load_books(file_path): 
    try:
        df = pd.read_csv(file_path)
        columns_to_rename = {
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Image-URL-L': 'image_url'
        }
        existing_columns_to_rename = {
            old: new for old, new in columns_to_rename.items() if old in df.columns
        }
        df = df.rename(columns=existing_columns_to_rename)

        if 'title' not in df.columns or 'ISBN' not in df.columns:
            st.error("Kolom 'title' atau 'ISBN' tidak ditemukan di data buku setelah rename.")
            return pd.DataFrame()

        df.dropna(subset=['title'], inplace=True)
        df['title'] = df['title'].astype(str)
        if 'author' in df.columns:
            df['author'] = df['author'].astype(str).fillna("Penulis tidak diketahui")
        if 'image_url' in df.columns:
            df['image_url'] = df['image_url'].astype(str).fillna("")

        print(f"Berhasil memuat dan memproses: {file_path}.")
        return df
    except FileNotFoundError:
        st.error(f"Berkas data buku ({file_path}) tidak ditemukan.")
    except Exception as e:
        st.error(f"Error memuat data buku dari {file_path}: {e}")
    return pd.DataFrame()

def load_recommender_model_as_tfsmlayer(model_directory_path: str, call_endpoint: str = 'serving_default'):
    try:
        model_layer = keras.layers.TFSMLayer(model_directory_path, call_endpoint=call_endpoint)
        print(f"TFSMLayer dari {model_directory_path} berhasil diinisialisasi.")
        return model_layer
    except Exception as e:
        error_message = f"Gagal menginisialisasi TFSMLayer dari {model_directory_path}. Error: {e}"
        print(error_message)
        import traceback
        print(traceback.format_exc())
        st.error(error_message)
        st.text_area("Detail Error Model CF", traceback.format_exc(), height=200)
    return None
