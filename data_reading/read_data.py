import pandas as pd
from data_reading.preprocess_data import apply_lower_case, remove_punctuation, \
    apply_lemmatization, apply_stemming, apply_strip
import os
import pickle
from functools import lru_cache

PICKLED_FEATURES_PATH = os.path.dirname(__file__) + "/../data/pickled_features/"
PICKLED_PPDB = os.path.dirname(__file__) + "/../data/ppdb/"


def read_clean_dataset():
    """Returns the dataset as provided by the author"""
    dataset = pd.read_csv(os.path.dirname(__file__) + "/../data/url-versions-2015-06-14-clean.csv")
    return dataset


@lru_cache(maxsize=1000000)
def read_ppdb_data():
    with open(os.path.join(PICKLED_PPDB, 'ppdb.pickle'), 'rb') as f:
        return pickle.load(f)


def read_pickle_file(filename: str) -> pd.DataFrame:
    """Returns the components of a given pickle file"""
    if '.pkl' not in filename:
        filename += '.pkl'
    with open(os.path.join(PICKLED_FEATURES_PATH, filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    clean_data = read_clean_dataset()

    # Remove unused columns
    clean_data = clean_data.drop(columns=['articleId', 'claimId'])

    # Lower case text
    clean_data = apply_lower_case(clean_data)

    # Remove punctuation apart from question marks - needed for feature extraction
    clean_data = remove_punctuation(clean_data)

    # Apply lemmatization
    clean_data = apply_lemmatization(clean_data)

    # Apply stemming
    clean_data = apply_stemming(clean_data)

    # Strip the data
    clean_data = apply_strip(clean_data)

    # Serialize data
    clean_data.to_pickle("../data/dummy.pkl")
