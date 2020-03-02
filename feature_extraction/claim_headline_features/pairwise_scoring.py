import numpy as np
import pandas as pd
import nltk
from data_reading.read_data import read_ppdb_data
from munkres import Munkres, make_cost_matrix

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH
from data_reading.preprocess_data import apply_lower_case, apply_strip

_max_ppdb_score = 10.0
_min_ppdb_score = -_max_ppdb_score

_munk = Munkres()


def compute_paraphrase_score(s, t):
    """Return numerical estimate of whether t is a paraphrase of s, up to
    stemming of s and t."""
    if s == t:
        return _max_ppdb_score

    # get PPDB paraphrases of s, and find matches to t, up to stemming
    s_paraphrases = set(read_ppdb_data().get(s, []))
    matches = set(filter(lambda x: x[0] == t, s_paraphrases))
    if matches:
        return max(matches, key=lambda x: x[1])[1]
    return _min_ppdb_score


def calc_hungarian_alignment_score(claim, headline):
    """Calculate the alignment score between the two texts s and t
    using the implementation of the Hungarian alignment algorithm
    provided in https://pypi.python.org/pypi/munkres/."""
    claim_tokens = nltk.word_tokenize(claim)
    headline_tokens = nltk.word_tokenize(headline)

    df = pd.DataFrame(index=claim_tokens, columns=headline_tokens, data=0.)

    for c in claim_tokens:
        for a in headline_tokens:
            df.loc[c, a] = compute_paraphrase_score(c, a)

    matrix = df.values
    cost_matrix = make_cost_matrix(matrix, lambda cost: _max_ppdb_score - cost)

    indices = _munk.compute(cost_matrix)
    total = 0.0
    for row, column in indices:
        value = matrix[row][column]
        total += value

    # Divide total revenue by size of lower dimension (n) (no. of tokens in shorter claim/headline)
    # to normalize since the algorithm will always return n pairs of indices.
    return total / float(np.min(matrix.shape))


def apply_kuhn_munkres(df):
    d = pd.DataFrame()
    scores = []
    for claim, headline in zip(df.claimHeadline, df.articleHeadline):
        scores.append(calc_hungarian_alignment_score(claim, headline))

    d['Kuhn-Munkres'] = scores
    return d


dataset = read_clean_dataset()  # Read the dataset
dataset = apply_lower_case(dataset)
dataset = apply_strip(dataset)

a = apply_kuhn_munkres(dataset)

a.to_pickle(PICKLED_FEATURES_PATH+"kuhn_munkres.pkl")