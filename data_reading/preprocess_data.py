import string
from data_reading.utils import entry_stem, entry_lemma, currenciate, clean_non_alphanumeric, numbers_to_words


def apply_lower_case(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(str.lower)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(str.lower)
    return dataframe


def apply_strip(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(str.strip)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(str.strip)
    return dataframe


def remove_punctuation(dataframe):
    punctuations = string.punctuation.replace('?', '')
    regexp = '[{}]'.format(punctuations)
    dataframe['claimHeadline'] = dataframe['claimHeadline'].str.replace(regexp, '')
    dataframe['articleHeadline'] = dataframe['articleHeadline'].str.replace(regexp, '')
    return dataframe


def apply_lemmatization(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(entry_lemma)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(entry_lemma)
    return dataframe


def apply_stemming(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(entry_stem)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(entry_stem)
    return dataframe


def remove_non_alphanumeric(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(clean_non_alphanumeric)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(clean_non_alphanumeric)
    return dataframe


def translate_currencies(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(currenciate)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(currenciate)
    return dataframe


def translate_numbers_to_words(dataframe):
    dataframe['claimHeadline'] = dataframe['claimHeadline'].apply(numbers_to_words)
    dataframe['articleHeadline'] = dataframe['articleHeadline'].apply(numbers_to_words)
    return dataframe
