import Model
import pandas as pd
from feature_extraction.headline_features import bag_of_words, root_dist_extractor


def extract_features(data):
    data = add_root_dist_feature(data)


def add_bag_of_words_features(data):
    return None


def add_root_dist_feature(data):
    return root_dist_extractor.extract_root_dist(data)
