import pandas as pd
from data_reading.preprocess_data import apply_lower_case, remove_punctuation, \
    apply_lemmatization, apply_stemming, apply_strip


def read_clean_dataset():
    """Returns the dataset as provided by the author"""
    dataset = pd.read_csv("../data/url-versions-2015-06-14-clean.csv")
    return dataset


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

