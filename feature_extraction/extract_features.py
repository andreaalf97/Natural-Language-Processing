import Model
from feature_extraction.headline_features import bag_of_words


def extract_features(data):
    model = Model()
    add_bag_of_words_features(model, data)


def add_bag_of_words_features(model, entry):
    model.features.extend(bag_of_words.calculate_common_words_features(entry))