import re
import pandas as pd
import numpy as np
import emoji
import os
import json

from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nlp_id.lemmatizer import Lemmatizer

path = os.getcwd()
resources_folder = 'resources'
resources_path = os.path.join(path, resources_folder)
model_folder = 'models'
model_path = os.path.join(path, model_folder)

with open(os.path.join(resources_folder,'slang_word_dictionary.json'), 'r') as json_file:
    SLANGWORD_DICT = json.load(json_file)

with open(os.path.join(resources_folder,'emoji_dictionary.json'), 'r') as json_file:
    EMOJI_DICT = json.load(json_file)

additional_stopwords = [
    'sih', 'nya', 'iya', 'pak', 'se', 'ya', 'ke', 'pa', 'om', 'hmmmmm', 'hmm', 'oh', 'eh', 'com'
]
excluded_stopwords = [
    'tidak', 'belum', 'bukan', 'tanpa', 'jarang', 'kurang', 'hampir tidak', 'tidak pernah',
    'belum pernah', 'tidak boleh', 'tidak bisa', 'tidak seharusnya', 'tidak mungkin',
    'tidak akan', 'tidak harus', 'tidak mengizinkan', 'tidak diizinkan', 'tidak diinginkan',
    'tidak disarankan', 'tidak disetujui', 'baik', 'bisa', 'mungkin', 'boleh', 'salah', 'semakin',
    'sangat', 'suka'
]
stop_words = set(stopwords.words('indonesian'))
stop_words_excluded = [value for value in stop_words if value not in excluded_stopwords] + additional_stopwords
stop_words_all = list(stop_words) + additional_stopwords


# Text processing function

def clean_tweet(text):
    cleaned_text = re.sub(r"#\w+", "", text)
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)  # Remove URLs starting with http/https
    cleaned_text = re.sub(r'www\.[^ ]+', '', cleaned_text)  # Remove www URLs
    cleaned_text = re.sub(r'pic\.twitter\.com/\S+', '', cleaned_text)  # Remove pic.twitter.com links
    cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_text)  # Remove emails
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text) # Remove HTML
    cleaned_text = cleaned_text.replace('@', '')  # Remove @ in mentions

    return cleaned_text

def remove_emojis(text):
    emojis_to_remove = ["😂", "…", "👍", "😁", '😄','😆', '😊', '😭']
    for emoji in emojis_to_remove:
        text = text.replace(emoji, "")
    return text

def extract_emojis(text):
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    combined_emoji = ''.join(emojis)
    return combined_emoji

def process_punctuation(text):
    modified_text = re.sub(r'[-\']', '',text)
    return modified_text

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    cleaned_text = re.sub(punctuation_pattern, ' ', text)
    return cleaned_text

def remove_extra_spaces(text):
    extra_spaces_pattern = r"\s+"
    cleaned_text = re.sub(extra_spaces_pattern, " ", text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def remove_special_characters(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return cleaned_text

def lowercase(text):
    lowercase_text = text.lower()
    return lowercase_text

def process_slang(text, dictionary=SLANGWORD_DICT):
    pattern_short_words = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in dictionary.keys()) + r')\b')
    output_text = pattern_short_words.sub(lambda x: dictionary[x.group()], text)
    return output_text

def remove_stopwords(text, stopwords=stop_words_excluded):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words_excluded]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def remove_specific_numbers(text):
    pattern = re.compile(r'\b(?!2\b|1\b|2019\b)\d+\b')
    result = re.sub(pattern, '', text)
    return result

# Text normalization function
lemmatizer = Lemmatizer()

def custom_lemmatization(text, exclude_words=[]):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) if word.lower() not in exclude_words else word for word in words]
    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text

def get_english_words(word_list):
    english_words = set(words.words())
    return [word for word in word_list if word.lower() in english_words]

# Word embedding
def mean_vector_df_val(model, train_sentences):
    vectorized_lists_train = []
    for word_list in train_sentences:
        vectorized_words_train = [model.wv[word] for word in word_list if word in model.wv]  # OOV handler (ignore)
        vectorized_lists_train.append(vectorized_words_train)

    mean_vector_train = []
    for matrix in vectorized_lists_train:
        transposed_matrix_train = list(map(list, zip(*matrix)))
        column_means_train = [sum(column) / len(column) for column in transposed_matrix_train]
        mean_vector_train.append(column_means_train)

    X_train_mean_vector = pd.DataFrame(np.array(mean_vector_train))

    return X_train_mean_vector