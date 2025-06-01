import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import ops

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_book, embedding_size, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_book = num_book
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        self.user_embedding = layers.Embedding( # layer embedding user
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias

        self.book_embedding = layers.Embedding( # layer embedding book_title
            num_book,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.book_bias = layers.Embedding(num_book, 1) # layer embedding book bias
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0]) # memanggil layer embedding 1
        user_vector = self.dropout(user_vector)
        user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2

        book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
        book_vector = self.dropout(book_vector)
        book_bias = self.book_bias(inputs[:, 1]) # memanggil layer embedding 4

        dot_user_book = ops.tensordot(user_vector, book_vector, 2) # perkalian dot product
        x = dot_user_book + user_bias + book_bias
        return ops.nn.sigmoid(x) # activation sigmoid

    def get_config(self):
        # Metode ini menyimpan konfigurasi yang diperlukan untuk membuat ulang layer.
        config = super().get_config() 
        config.update({
            "num_users": self.num_users,
            "num_book": self.num_book,
            "embedding_size": self.embedding_size,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
