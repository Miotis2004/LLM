import random
from tokenizer import tokenize, to_lowercase, remove_special_characters, remove_punctuation, create_vocabulary, numerical_encoding, pad_sequence

# Read the data from text.txt
with open("text.txt", 'r', encoding='utf-8') as file:
    text_sample = file.read()

tokens = tokenize(text_sample)
tokens = to_lowercase(tokens)
tokens = remove_special_characters(tokens)
tokens = remove_punctuation(tokens)
vocab = create_vocabulary(tokens)
encoded_tokens = numerical_encoding(tokens, vocab)

# Determine the split index
split_index = int(0.8 * len(encoded_tokens))

# Split the encoded tokens
train_data = encoded_tokens[:split_index]
validation_data = encoded_tokens[split_index:]

with open("train_data.txt", 'w', encoding='utf-8') as file:
    for item in train_data:
        file.write("%s\n" % item)

with open("validation_data.txt", 'w', encoding='utf-8') as file:
    for item in validation_data:
        file.write("%s\n" % item)
