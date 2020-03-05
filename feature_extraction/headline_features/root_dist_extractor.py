"""Computes the distance from hedging/refuting word to the root of the sentence
for all the headlines in the dataset"""

import stanfordnlp
#stanfordnlp.download('en')
import networkx as nx

from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH
from data_reading.preprocess_data import apply_lower_case, remove_punctuation, apply_strip


# List of words taken from github repo of the paper
_refuting_seed_words = [
                        'fake',
                        'fraud',
                        'hoax',
                        'false',
                        'deny', 'denies',
                        'refute',
                        'not',
                        'despite',
                        'nope',
                        'doubt', 'doubts',
                        'bogus',
                        'debunk',
                        'pranks',
                        'retract'
]

_refuting_words = _refuting_seed_words


_hedging_seed_words = [
        'alleged', 'allegedly',
        'apparently',
        'appear', 'appears',
        'claim', 'claims',
        'could',
        'evidently',
        'largely',
        'likely',
        'mainly',
        'may', 'maybe', 'might',
        'mostly',
        'perhaps',
        'presumably',
        'probably',
        'purported', 'purportedly',
        'reported', 'reportedly',
        'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
        'says',
        'seem',
        'somewhat',
        'supposedly',
        'unconfirmed']

_hedging_words = _hedging_seed_words

nlp = stanfordnlp.Pipeline()


def extract_root_dist(data):
    data['refute_dist'] = data['articleHeadline'].apply(extract_single_root_dist, words=_refuting_words)
    data['hedge_dist'] = data['articleHeadline'].apply(extract_single_root_dist, words=_hedging_words)
    return data


def extract_single_root_dist(entry, words):
    '''Returns mininum distance from hedhing/refuting words to the root of the sentence for whole headline (headline can have multiple sentences).'''
    doc = nlp(entry)
    min_dist = 0
    for sentence in doc.sentences:
        graph, root = create_dependency_graph(sentence)
        for word in words:
            if graph.has_node(word):
                min_dist = min(nx.shortest_path_length(graph, source=root, target=word), min_dist)
    return min_dist


# Dependency graph is a graph (tree) with words as nodes and if
# word A is dependent on word B in a sentence, then there is an edge from B to A
def create_dependency_graph(sentence):
    '''Creates dependency graph for the sentence using StanfordNLP'''
    edges = []
    root = ''
    for token in sentence.dependencies:
        dep = token[0].text.lower()
        if dep != 'root':
            edges.append((dep, token[2].text))
        else:
            root = token[2].text
    return nx.Graph(edges), root


dataset = read_clean_dataset()  # Read the dataset
dataset = apply_lower_case(dataset)
dataset = apply_strip(dataset)

dataset = extract_root_dist(dataset)

a = dataset[['refute_dist', 'hedge_dist']]
a.to_pickle(PICKLED_FEATURES_PATH+"root_dist.pkl")