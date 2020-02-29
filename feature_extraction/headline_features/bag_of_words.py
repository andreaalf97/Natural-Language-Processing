"""Creates the bag of words representation of all the headlines in the dataset
and whether they finish with a question mark"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np

from read_data import read_clean_dataset

from functools import lru_cache
from collections import Counter


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


dataset = read_clean_dataset() # Read the dataset

counts = create_corpus(dataset) # Number of occurrences of each word in the corpus
assignments = dict(zip(counts.keys(), range(len(counts)))) # Index of each of the words in the vector
print(counts)
d = create_vectors(dataset, assignments) # dataframe with all the vectors


pickle_path = "../data/pickled_features/bow.pickle"
d.to_pickle(pickle_path) # pickle the dataframe to the specified folder
