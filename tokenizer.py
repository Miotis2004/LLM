import re
import nltk
from collections import Counter
nltk.download('punkt')

# 4.2.1. Tokenization
def tokenize(text):
    return nltk.word_tokenize(text)

# ... [rest of the preprocessing functions remain unchanged]

# Point to the file and read its content
file_path = "text.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text_sample = file.read()

# Tokenization
tokens = tokenize(text_sample)
print(f"Tokens: {tokens}")

# 4.2.2. Lowercasing
def to_lowercase(tokens):
    return [token.lower() for token in tokens]

# 4.2.3. Remove Special Characters and Clean Data
def remove_special_characters(tokens):
    # A regex pattern to match any token that's not an alphanumeric (word or number)
    pattern = re.compile(r'[^a-z0-9\s]')
    return [token for token in tokens if not pattern.match(token)]

# 4.2.4. Handle Punctuation (assuming removal in this case)
def remove_punctuation(tokens):
    return [token for token in tokens if token.isalnum()]

# 4.2.6. Vocabulary Creation
def create_vocabulary(tokens):
    # Count the frequency of each token
    freq = Counter(tokens)
    # Sort tokens by frequency (from high to low)
    sorted_vocab = sorted(freq, key=freq.get, reverse=True)
    # Assign an ID to each token (starting from 1 for simplicity; 0 will be reserved for padding)
    vocab = {word: index+1 for index, word in enumerate(sorted_vocab)}
    return vocab

# 4.2.7. Numerical Encoding
def numerical_encoding(tokens, vocab):
    return [vocab.get(token, 0) for token in tokens]  # 0 is used for tokens not found in the vocab

# 4.2.8. Padding Sequences
def pad_sequence(encoded_seq, max_length=20):
    return encoded_seq[:max_length] + [0]*(max_length - len(encoded_seq))

# Sample text for demonstration
#text_sample = "Hello! How are you doing today? I'm doing well."

# Tokenization
tokens = tokenize(text_sample)
print(f"Tokens: {tokens}")

# Lowercasing
tokens = to_lowercase(tokens)
print(f"Lowercased Tokens: {tokens}")

# Remove Special Characters
tokens = remove_special_characters(tokens)
print(f"Tokens after removing special characters: {tokens}")

# Remove Punctuation
tokens = remove_punctuation(tokens)
print(f"Tokens after removing punctuation: {tokens}")

# Create Vocabulary
vocab = create_vocabulary(tokens)
print(f"Vocabulary: {vocab}")

# Numerical Encoding
encoded_tokens = numerical_encoding(tokens, vocab)
print(f"Numerically Encoded Tokens: {encoded_tokens}")

# Padding Sequences
padded_sequence = pad_sequence(encoded_tokens)
print(f"Padded Sequence: {padded_sequence}")
