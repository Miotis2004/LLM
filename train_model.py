import numpy as np
import tensorflow as tf
import os  # Added import for 'os' as it is used for path operations
from tokenizer import tokenize, to_lowercase, remove_special_characters, remove_punctuation, create_vocabulary, numerical_encoding
from model import build_model, compile_model  # We'll import compile_model

# Set the device to GPU
with tf.device('/device:GPU:0'):
    # Load training data from train_data.txt
    with open("train_data.txt", 'r', encoding='utf-8') as file:
        train_data = [int(line.strip()) for line in file.readlines()]

    # Compute the vocabulary size
    vocab_size = max(train_data) + 1

    '''# Create training sequences and their corresponding labels
    SEQUENCE_LENGTH = 100
    sequences = []
    labels = []

    for i in range(0, len(train_data) - SEQUENCE_LENGTH, 1):
        sequences.append(train_data[i:i + SEQUENCE_LENGTH])
        labels.append(train_data[i + SEQUENCE_LENGTH])

    # Convert sequences and labels into numpy arrays for TensorFlow
    train_sequences = np.array(sequences)
    train_labels = np.array(labels)'''

    # Create training sequences and their corresponding labels
    SEQUENCE_LENGTH = 100
    sequences = []
    next_sequences = []

    for i in range(0, len(train_data) - SEQUENCE_LENGTH - 1, 1):
        sequences.append(train_data[i:i + SEQUENCE_LENGTH])
        next_sequences.append(train_data[i + 1:i + SEQUENCE_LENGTH + 1])

    # Convert sequences and labels into numpy arrays for TensorFlow
    train_sequences = np.array(sequences)
    train_labels = np.array(next_sequences)



    # Hyperparameters
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024
    BATCH_SIZE = 64

    model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

    # Use the compile_model function from model.py
    model = compile_model(model) 

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    EPOCHS = 10
    history = model.fit(train_sequences, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint_callback])

    model.save('text_generation_model.h5')
