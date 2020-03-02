"""Extract Subject-Verb-Object triplets from headlines and their corresponding claim, and
extract whether they're paraphrashes using PPDB"""

import pandas as pd
import spacy
from textacy.extract import subject_verb_object_triples
from spacy.symbols import nsubj, VERB, dobj
from spacy import displacy
import numpy as np

from functools import lru_cache

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH, read_ppdb_data, read_pickle_file


@lru_cache(maxsize=50)
def find_possible_svo_triples(sentence: str, nlp) -> (set, set, set):
    headline = nlp(sentence)
    s = []
    v = []
    o = []
    for possible_subject in headline:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            # For more examples see https://spacy.io/usage/linguistic-features#dependency-parse
            # We found a verb
            # add verb
            v.append(possible_subject.head)
            # # add subject
            # s.append(possible_subject)
            # # add object
            # o.append([child for child in possible_subject.head.children if child.dep == dobj])
    return s, v, o


def extract_svo(column, nlp):
    subjects = []
    verbs = []
    objects = []
    i = 0
    for headline in column:
        # print(headline)
        triplet = subject_verb_object_triples(nlp(headline))
        try:
            s, v, o = next(triplet)
        except Exception:
            s = None
            v = None
            o = None
        subjects.append(s)
        verbs.append(v)
        objects.append(o)
        i += 1
        print(f'{headline} -> Verb = {v}, Subject = {s}, Objects = {o}')
    return subjects, verbs, objects


def create_svo_triplets():
    print('Loading vectors')
    # previous to this you have to download the model with
    # python -m spacy download en_core_web_md
    nlp = spacy.load('en_core_web_md')

    print('Vectors loaded')
    dataset = read_clean_dataset()
    s, v, o = extract_svo(dataset.articleHeadline, nlp)

    # Have to convert everything to string
    s = [str(n) for n in s]
    v = [str(n) for n in v]
    o = [str(n) for n in o]
    p = pd.DataFrame()

    p['subjects'] = s
    p['verbs'] = v
    p['objects'] = o
    p.to_pickle(PICKLED_FEATURES_PATH + 'claimHeadlinesSVO.pkl')


# entailment scores for the ppdb stuff
_entailment_map = {
    'ReverseEntailment': 0,
    'ForwardEntailment': 1,
    'Equivalence': 2,
    'OtherRelated': 2,
    'Independence': 3,
    'NotFound': 4
}

ppdb = read_ppdb_data()


@lru_cache(maxsize=10000)
def _calculate_entailment(h, c) -> np.array:
    """Calculates entailment between the given words of headline (h) and the claim (c)

    The funtion returns a vector with the corresponding component set to 1

    vec = ('ReverseEntailment', 'ForwardEntailment', 'Equivalence', 'OtherRelated', 'Independence', 'NotFound')
    """
    # Construct the vector
    v = np.zeros((1, len(set(_entailment_map.values()))))

    # If one of the components of the triplet comparison was not found
    if h == 'None' or c == 'None':
        v[0, _entailment_map['NotFound']] = 1
        return v

    # If the terms are the same
    if h.lower() == c.lower():
        v[0, _entailment_map['Equivalence']] = 1
        return v

    # If there's other relationship we have to analyze with PPDB
    relationships = [(x, s, e) for (x, s, e) in ppdb.get(h, [])
                     if e in _entailment_map.keys() and x == c]

    # Return the relationship with the max score
    if relationships:
        relationship = max(relationships, key=lambda t: t[1])[2]
        v[0, _entailment_map[relationship]] = 1

    return v


def compute_similarities() -> np.array:
    """ Returns a matrix of shape (n_headlines, 15),
    with those 15 columns being the concatenation of the entailment vectors

    ENTAILMENT_SUBJECT (1x5) | ENTAILMENT_VERB (1x5) | ENTAILMENT_OBJECT (1x5)

    with each of those being a one-hot vector like

    vec = ('ReverseEntailment', 'ForwardEntailment', 'Equivalence', 'OtherRelated', 'Independence', 'NotFound')
    """

    headline_svo = read_pickle_file("articleHeadlinesSVO")
    claim_svo = read_pickle_file("claimHeadlinesSVO")

    # create the matrix that will keep track of the entailments
    mat = np.zeros((len(headline_svo), 3 * len(set(_entailment_map.values()))))
    print(mat.shape)
    assert mat.shape == (2595, 15)

    # Iterate through all the columns
    for i, (sh, vh, oh, sc, vc, oc) in enumerate(zip(headline_svo.subjects, headline_svo.verbs, headline_svo.objects,
                                                     claim_svo.subjects, claim_svo.verbs, claim_svo.objects)):
        print(f'{sh, vh, oh} -- VS -- {sc, vc, oc}')
        # Vector with the entailments of a headline-claim pairing
        vec = np.zeros((1, 3 * len(set(_entailment_map.values()))))

        # Get the similarity between the subjects
        subject_entailment = _calculate_entailment(sh, sc)
        # Get the similarity between the verbs
        verb_entailment = _calculate_entailment(vh, vc)
        # Get the similarity between the objects
        object_entailment = _calculate_entailment(oh, oc)

        # Add them to the matrix
        vec[0, 0:5] = subject_entailment
        vec[0, 5:10] = verb_entailment
        vec[0, 10:15] = object_entailment

        mat[i, :] = vec

    return mat


m = compute_similarities()
d = pd.DataFrame(m)

d.to_pickle(PICKLED_FEATURES_PATH + "SVO.pkl")
