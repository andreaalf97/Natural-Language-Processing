from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def entry_stem(entry):
    tokenized_entry = word_tokenize(entry)
    stemmer = PorterStemmer()
    stemmed_entry = ""
    for word in tokenized_entry:
        stemmed_entry = stemmed_entry + stemmer.stem(str(word)) + ' '
    return stemmed_entry


def entry_lemma(entry):
    tokenized_entry = word_tokenize(entry)
    lemmatizer = WordNetLemmatizer()
    lemmatized_entry = ""
    for word in tokenized_entry:
        lemmatized_entry = lemmatized_entry + lemmatizer.lemmatize(str(word)) + ' '
    return lemmatized_entry


