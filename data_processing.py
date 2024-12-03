#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import re
import joblib
from wordcloud import WordCloud
from tqdm import tqdm

def expand_contractions(text):
    contraction_patterns = [(r'can\'t', 'cannot'),
                            (r'won\'t', 'will not'),
                            # Add more patterns as needed
                           ]
    for pattern, replacement in contraction_patterns:
        text = re.sub(pattern, replacement, text)
    return text


# # Function to remove URLs and mentions

# In[6]:


def remove_urls_mentions(text):
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    return text


# # Function to remove special characters

# In[7]:


def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# # Define a function to preprocess a single text

# In[9]:


def preprocess_text(text):
    # Load the spaCy English language model
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    
    # Convert the text to lowercase
    text = text.lower()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove URLs and mentions
    text = remove_urls_mentions(text)
    
    # Remove special characters
    text = remove_special_characters(text)
    
    # Tokenize the text using spaCy
    doc = nlp(text)
    
    # Lemmatize the tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Join the tokens back into a single string
    clean_text = ' '.join(tokens)
    
    return clean_text



