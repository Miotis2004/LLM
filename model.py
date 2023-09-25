import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = Sequential([
        # Embedding layer converts token IDs to vectors
        Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # LSTM layer with 'rnn_units' number of units.
        LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        # Dense (fully connected) layer that outputs one logit for each word in the vocabulary.
        Dense(vocab_size)
    ])
    return model

# Compile the Model
def compile_model(model):
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer='adam', loss=loss)
    return model

# If you want to execute the below code for testing purposes, do it under this guard:
if __name__ == "__main__":
    # Hyperparameters (you can tune these)
    VOCAB_SIZE = 5000  # This is just a dummy value for testing purposes; replace it with your actual vocab size.
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    BATCH_SIZE = 64

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model = compile_model(model)
    model.summary()
