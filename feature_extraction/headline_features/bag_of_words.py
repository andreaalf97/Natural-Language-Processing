"""Creates the bag of words representation of all the headlines in the dataset
and whether they finish with a question mark"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH
from data_reading.preprocess_data import apply_lower_case, apply_lemmatization, remove_non_alphanumeric

from functools import lru_cache
from collections import Counter

from typing import List


def create_bow(headlines: List, max_ngram_size=2, number_of_features=1000, remove_stopwords=True,
               apply_tfidf=True) -> (list, sparse.csr_matrix):
    """Creates the bag of words representation using the specialized sklearn functionality

    Also by default apply TF-IDF correction to the returned dataset, however it can be turned off
    in case we see that without TF-IDF shows better performance
    """
    sw = stopwords.words('english') if remove_stopwords else None
    if apply_tfidf:
        pipe = Pipeline([('count', CountVectorizer(ngram_range=(1, max_ngram_size), max_features=number_of_features,
                                                   stop_words=sw)),
                         ('tfidf', TfidfTransformer())])
        _bow = pipe.fit_transform(headlines)
        f = pipe['count'].get_feature_names()
    else:
        cv = CountVectorizer(ngram_range=(1, max_ngram_size), max_features=number_of_features, stop_words=sw)
        _bow = cv.fit_transform(headlines)
        f = cv.get_feature_names()
    return f, _bow


# Lemmatize the dataset for better representation
dataset = read_clean_dataset()  # Read the dataset
dataset = remove_non_alphanumeric(dataset)
dataset = apply_lower_case(dataset)
dataset = apply_lemmatization(dataset)

# get features without tfidf
features, b = create_bow(dataset.articleHeadline.values, apply_tfidf=False)

# create a dataset
d = pd.DataFrame(b.toarray())
d.columns = features
d.to_pickle(PICKLED_FEATURES_PATH + "bow.pkl")  # pickle the dataframe to the specified folder

# get features with tfidf
features, b = create_bow(dataset.articleHeadline.values, apply_tfidf=True)

# create a dataset
df = pd.DataFrame(b.toarray())
df.columns = features
df.to_pickle(PICKLED_FEATURES_PATH + "tfidf.pkl")  # pickle the dataframe to the specified folder
