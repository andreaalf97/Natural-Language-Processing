"""Compares the claim and the headline and returns the pairing score"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from scipy import spatial
from functools import lru_cache, reduce
import pandas as pd
import numpy as np

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH


def normalize_sentences(claim: str, headline: str, wnl: WordNetLemmatizer) -> (str, str):
    """Removes stopwords and lemmatizes both sentences"""
    claim = word_tokenize(claim.lower())
    headline = word_tokenize(headline.lower())
    # remove non alphanumeric tokens and stopwords
    claim = [w for w in claim if (w.isalpha() and w not in stopwords.words('english'))]
    headline = [w for w in headline if (w.isalpha() and w not in stopwords.words('english'))]
    # Remove stop
    claim = [wnl.lemmatize(w) for w in claim]
    headline = [wnl.lemmatize(w) for w in headline]
    return ' '.join(claim), ' '.join(headline)


@lru_cache(maxsize=100)
def compute_similarity(claim, headline, nlp):
    """ Computes the similarity of the word2vec representations of the claim and
    the given headline according to different criteria

    When computing similarity with the built-in spacy function, what it does internally
    is the average of the vectors of the words in the sentence
    """
    wnl = WordNetLemmatizer()
    claim, headline = normalize_sentences(claim, headline, wnl)
    # Calculate the vector as the average word vector
    claim_w2v = nlp(claim)
    headline_w2v = nlp(headline)
    # compute the average similarity
    avg_sim = claim_w2v.similarity(headline_w2v)
    # Calculate the vector as the product of word vectors
    # Have to take logs since the product vanishes due to small numbers
    claim_prod_vector = reduce(lambda x, y: x * y, [nlp(t).vector for t in claim])
    headline_prod_vector = reduce(lambda x, y: x * y, [nlp(t).vector for t in headline])
    # calculate cosine similarity
    prod_sim = 1 - spatial.distance.cosine(claim_prod_vector, headline_prod_vector)
    return avg_sim, prod_sim


def claim_to_headline_sim(d: pd.DataFrame, nlp) -> pd.DataFrame:
    """ Calculates the similarity of all the headlines and the corresponding claims"""
    # Empty similarities list
    avg_similarities = []
    prod_similarities = []
    i = 0
    for claim, headline in zip(d.claimHeadline, d.articleHeadline):
        avg_sim, prod_sim = compute_similarity(claim, headline, nlp)
        avg_similarities.append(avg_sim)
        prod_similarities.append(prod_sim)
        i += 1
        if i % 50 == 0:
            print(f'[{i}] Sim between {claim} ||||| {headline} --> ({avg_sim}/{prod_sim})')
    # After computing all similarities, add a new column to the dataframe
    d = pd.DataFrame()
    d['avg_similarity'] = avg_similarities
    d['prod_similarity'] = prod_similarities
    return d


if __name__ == '__main__':
    # Vector directory in my computer
    VECTOR_DIR = "../../../wse/vec"

    # Load the clean dataset
    df = read_clean_dataset()

    # Load the vectors (vectors are number 3 from https://fasttext.cc/docs/en/english-vectors.html)
    print('Loading vectors')
    nlp = spacy.load(VECTOR_DIR)

    print('Loaded vectors')
    similarity_df = claim_to_headline_sim(df, nlp)

    print('Saving features to', PICKLED_FEATURES_PATH)
    similarity_df.to_pickle(PICKLED_FEATURES_PATH + "word2vec.pkl")
