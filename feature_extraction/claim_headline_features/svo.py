"""Extract Subject-Verb-Object triplets from headlines and their corresponding claim, and
extract whether they're paraphrashes using PPDB"""

import pandas as pd
import spacy
import spacy.symbols as sym
from nltk.corpus import stopwords
from preprocess_data import remove_non_alphanumeric
import numpy as np
import re


from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH, read_ppdb_data, read_pickle_file

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
sw = stopwords.words('english')


def _calculate_entailment(h: list, c: list) -> np.array:
    """Calculates entailment between the given words of headline (h) and the claim (c)

    The funtion returns a vector with the corresponding component set to 1

    vec = ('ReverseEntailment', 'ForwardEntailment', 'Equivalence', 'OtherRelated', 'Independence', 'NotFound')
    """
    # Construct the vector
    v = np.zeros((1, len(set(_entailment_map.values()))))

    # Lowercase everything and get rid of stopwords
    c = [w.lower() for w in c if w not in sw]
    h = [w.lower() for w in h if w not in sw]

    # If one of the components of the triplet comparison was not found
    if len(h) == 0 or len(c) == 0:
        v[0, _entailment_map['NotFound']] = 1
        return v

    # Do a for loop here with the multiple terms (nested for??)
    # If the terms are the same
    for tok_h in h:
        for tok_c in c:
            # If they're equal
            if tok_h == tok_c:
                v[0, _entailment_map['Equivalence']] += 1
                continue
            # Other relationship
            relationships = [(x, s, e) for (x, s, e) in ppdb.get(tok_h, [])
                             if e in _entailment_map.keys() and x == tok_c]
            if relationships:
                relationship = max(relationships, key=lambda t: t[1])[2]
                v[0, _entailment_map[relationship]] += 1

    return v


def compute_similarities_ppdb() -> np.array:
    """ Returns a matrix of shape (n_headlines, 15),
    with those 15 columns being the concatenation of the entailment vectors

    ENTAILMENT_SUBJECT (1x5) | ENTAILMENT_VERB (1x5) | ENTAILMENT_OBJECT (1x5)

    with each of those being a one-hot vector like

    vec = ('ReverseEntailment', 'ForwardEntailment', 'Equivalence', 'OtherRelated', 'Independence', 'NotFound')
    """

    headline_svo = read_pickle_file("articleHeadlineSVO")
    claim_svo = read_pickle_file("claimHeadlineSVO")

    # create the matrix that will keep track of the entailments
    mat = np.zeros((len(headline_svo), 3 * len(set(_entailment_map.values()))))
    print(mat.shape)
    assert mat.shape == (2595, 15)

    # Iterate through all the columns
    for i, (sh, vh, oh, sc, vc, oc) in enumerate(zip(headline_svo.s, headline_svo.v, headline_svo.o,
                                                     claim_svo.s, claim_svo.v, claim_svo.o)):
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

    return headline_svo, claim_svo, mat


def _sim(h: list, c: list, nlp):
    """Returns the average similarity between all the terms in the arrays"""
    # remove stopwords and lowercase everything
    c = [w.lower() for w in c if w not in sw]
    h = [w.lower() for w in h if w not in sw]
    # Vector of similarities
    if len(h) == 0 or len(c) == 0:
        return 0
    sims = []
    for tok_h in h:
        for tok_c in c:
            a = nlp(tok_h)
            b = nlp(tok_c)
            sims.append(a.similarity(b))
    # Return the average similarity between words
    return np.mean(sims)


def compute_similarity_word2vec(nlp):
    """Return the w2v similarity between the vector representation of subjects, verbs and objects"""
    headline_svo = read_pickle_file("articleHeadlineSVO")
    claim_svo = read_pickle_file("claimHeadlineSVO")

    # create the matrix that will keep track of the entailments
    mat = np.zeros((len(headline_svo), 3))
    print(mat.shape)
    assert mat.shape == (2595, 3)

    # Iterate through all the columns
    for i, (sh, vh, oh, sc, vc, oc) in enumerate(zip(headline_svo.s, headline_svo.v, headline_svo.o,
                                                     claim_svo.s, claim_svo.v, claim_svo.o)):
        print(f'{sh, vh, oh} -- VS -- {sc, vc, oc}')
        # Vector with the entailments of a headline-claim pairing
        vec = np.zeros((1, 3))

        # Get the similarity between the subjects
        subject_sim = _sim(sh, sc, nlp)
        # Get the similarity between the verbs
        verb_sim = _sim(vh, vc, nlp)
        # Get the similarity between the objects
        object_sim = _sim(oh, oc, nlp)

        # Add them to the matrix
        vec[0, 0] = subject_sim
        vec[0, 1] = verb_sim
        vec[0, 2] = object_sim

        mat[i, :] = vec

    return mat


def _extract_triplets(sentence: str, nlp) -> (list, list, list):
    """Extract the triplets from a single sentence"""
    characters = [',', '.', '?', ';', ':']
    s = []
    v = []
    o = []
    # Split by commas or other signs
    sentences = re.split(r"([,?;.:])", sentence)
    for sent in sentences:
        # Ignore single punctuation characters
        if sent in characters:
            continue
        sent = sent.strip()
        sent = nlp(sent)
        # Extract the chunks of the sentence
        chunks = sent.noun_chunks
        for chunk in chunks:
            if chunk.root.dep == sym.nsubj or chunk.root.dep == sym.nsubjpass:
                s.append(chunk.root.text.strip())
                # Append the verb
                v.append(chunk.root.head.text.strip())
            elif chunk.root.dep == sym.dobj or chunk.root.dep == sym.pobj:
                o.append(chunk.root.text.strip())

    return s, v, o


def svo_extraction(column, nlp):
    subjects = []
    verbs = []
    objects = []

    for headline in column:
        s, v, o = _extract_triplets(headline, nlp)
        print(f'{headline} -> S = {s}, V = {v}, O = {o}')
        subjects.append(s)
        objects.append(o)
        verbs.append(v)

    # Create dataframe
    df = pd.DataFrame()
    df = df.astype(object)
    df['s'] = subjects
    df['v'] = verbs
    df['o'] = objects
    return df


if __name__ == '__main__':
    print("Loading Vectors...")
    nlp = spacy.load('en_core_web_md')

    # Proper dataset
    d = read_clean_dataset()
    d = remove_non_alphanumeric(d)

    # Extract SVO triplets
    headlines = svo_extraction(d.articleHeadline, nlp)
    claims = svo_extraction(d.claimHeadline, nlp)

    # Save SVOs
    headlines.to_pickle(PICKLED_FEATURES_PATH + "articleHeadlineSVO.pkl")
    claims.to_pickle(PICKLED_FEATURES_PATH + "claimHeadlineSVO.pkl")

    # Extract entailments
    mat = compute_similarities_ppdb()

    VECTOR_DIR = "../../../wse/vec"
    print("Loading w2v vectors")
    nlp = spacy.load(VECTOR_DIR)
    mat = compute_similarity_word2vec(nlp)
