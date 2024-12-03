import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def split_data(file_path):
    """
    Preprocess the dataset for text classification.

    Args:
        file_path (str): Path to the dataset file (CSV format).

    Returns:
        tuple: Preprocessed training and testing data (X_train, X_test, y_train, y_test).
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Split into input and target
    X = df['clean_tweet'].values
    y = df['class'].values

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    vocab_size = len(tokenizer.word_index) + 1
    X = tokenizer.texts_to_sequences(X)

    # Pad sequences
    max_length = max(len(sequence) for sequence in X)
    X = pad_sequences(X, maxlen=max_length)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test,tokenizer
