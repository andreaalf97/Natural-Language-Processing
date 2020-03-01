"""Return two features:

1) If the headline ends with an ? sign
2) If the headline contains a ? sign

Just to see which one ends up performing better during tests

"""
import pandas as pd

from data_reading.read_data import PICKLED_FEATURES_PATH, read_clean_dataset


def extract_features(headlines):
    """Extract both features"""
    ends_with = []
    contains = []
    for headline in headlines:
        if '?' in headline:
            contains.append(True)
            if headline.endswith('?'):
                ends_with.append(True)
            else:
                ends_with.append(False)
        else:
            contains.append(False)
            ends_with.append(False)

    # After all the iteration create a dataframe with those two columns
    d = pd.DataFrame()
    d['q_ends'] = ends_with
    d['q_contains'] = contains
    return d


if __name__ == '__main__':
    dataset = read_clean_dataset()
    q_features = extract_features(dataset.articleHeadline)

    # save to the file
    q_features.to_pickle(PICKLED_FEATURES_PATH+"q_features.pkl")

