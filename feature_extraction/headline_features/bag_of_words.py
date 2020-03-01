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
from data_reading.preprocess_data import apply_lower_case, apply_lemmatization

from functools import lru_cache
from collections import Counter

from typing import List


@lru_cache(maxsize=50)
def bow(headline: str, wnl: WordNetLemmatizer) -> Counter:
    """Returns the frequency counter"""
    tokens = word_tokenize(headline.lower())
    # Remove stops and symbols
    tokens = [t for t in tokens if t.isalpha()]
    # Lemmatize the tokens
    tokens = [wnl.lemmatize(t) for t in tokens]
    counts = Counter(tokens)
    return counts


def create_corpus(df: pd.DataFrame) -> Counter:
    """Creates the corpus of all the words while summing the counters"""
    c = Counter()
    wnl = WordNetLemmatizer()
    i = 0
    for headline in df.articleHeadline:
        _c = bow(headline, wnl)
        c += _c
        i += 1
        if i % 500 == 0:
            print(c)
    return c


def vectorize(tokens, indexes: dict) -> np.array:
    """Convert the tokens to frequency vectors"""
    vec = np.zeros(len(indexes))
    for t in tokens:
        vec[indexes[t]] += 1
    return vec


def create_vectors(df: pd.DataFrame, indexes: dict) -> pd.DataFrame:
    """Create a dataframe with the BoW representations of the headers"""
    features = pd.DataFrame(columns=range(len(indexes)))
    wnl = WordNetLemmatizer()
    i = 0
    # put some elements to 1
    for headline, stance in zip(df.articleHeadline, df.articleHeadlineStance):
        if i % 100 == 0:
            print(i)
        # headline = row.articleHeadline
        tokens = word_tokenize(headline.lower())
        # Remove stops and symbols
        tokens = [t for t in tokens if t.isalpha()]
        # Lemmatize
        tokens = [wnl.lemmatize(t) for t in tokens]
        # Initialize vectors
        vec = vectorize(tokens, indexes)
        features.loc[i] = vec.tolist()
        i += 1
    return features


def create_bow(headlines: List, max_ngram_size=2, number_of_features=500, remove_stopwords=False,
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
dataset = apply_lower_case(dataset)
dataset = apply_lemmatization(dataset)

features, b = create_bow(dataset.articleHeadline.values)

# create a dataset
d = pd.DataFrame(b.toarray())
d.columns = features

# This is the old code to make use of the old bow representation with no optimizations or tf-idf
# counts = create_corpus(dataset)  # Number of occurrences of each word in the corpus
# assignments = dict(zip(counts.keys(), range(len(counts))))  # Index of each of the words in the vector
# print(counts)
# d = create_vectors(dataset, assignments)  # dataframe with all the vectors
#

d.to_pickle(PICKLED_FEATURES_PATH+"bow.pkl")  # pickle the dataframe to the specified folder
