import os

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np


# Code taken from William Ferreira's original MSc Project https://github.com/willferreira/mscproject
# Specifically from the file
# https://github.com/willferreira/mscproject/blob/deb92fba51e027c155899291767b55aeaeec45b6/bin/run_extract_ppdb_data.py

def to_float(s):
    return np.nan if s == 'NA' else float(s)

def process_ppdb_data():
    with open(os.path.join('..', '..', 'data', 'ppdb', 'ppdb-2.0-xl-all', 'ppdb-2.0-xl-all'), 'r') as f:
        ppdb = {}
        for line in f:
            data = line.split('|||')
            text_lhs = data[1].strip(' ,')
            text_rhs = data[2].strip(' ,')
            if len(text_lhs.split(' ')) > 1 or len(text_rhs.split(' ')) > 1:
                continue
            ppdb_score = to_float(data[3].strip().split()[0].split('=')[1])
            entailment = data[-1].strip()
            paraphrases = ppdb.setdefault(text_lhs, list())
            paraphrases.append((text_rhs, ppdb_score, entailment))
    return ppdb

# ------------------------------------------------------
#               End of code by willferreira
# ------------------------------------------------------


if __name__ == '__main__':
    ppdb = process_ppdb_data()

    with open(os.path.join('..', '..', 'data', 'ppdb', 'ppdb_xl.pickle'),'wb') as f:
        pickle.dump(ppdb, f, pickle.HIGHEST_PROTOCOL)