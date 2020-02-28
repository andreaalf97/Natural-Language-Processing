import pandas as pd

def read_clean_dataset():
    """Returns the dataset as provided by the author"""
    dataset = pd.read_csv("../data/url-versions-2015-06-14-clean.csv")
    return dataset