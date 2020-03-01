"""Extract Subject-Verb-Object triplets from headlines and their corresponding claim, and
extract whether they're paraphrashes using PPDB"""

import pandas as pd
import spacy
from textacy.extract import subject_verb_object_triples
from spacy.symbols import nsubj, VERB, dobj
from spacy import displacy

from functools import lru_cache

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH


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


def extract_svo(d: pd.DataFrame, nlp):
    subjects = []
    verbs = []
    objects = []
    i = 0
    for headline in d.claimHeadline:
        #print(headline)
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





print('Loading vectors')
# previous to this you have to download the model with
# python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

print('Vectors loaded')
dataset = read_clean_dataset()
s, v, o = extract_svo(dataset, nlp)

# Have to convert everything to string
s = [str(n) for n in s]
v = [str(n) for n in v]
o = [str(n) for n in o]
p = pd.DataFrame()

p['subjects'] = s
p['verbs'] = v
p['objects'] = o
p.to_pickle(PICKLED_FEATURES_PATH+'claimHeadlinesSVO.pkl')
