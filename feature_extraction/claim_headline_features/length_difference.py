from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH
import numpy as np
import pandas as pd

d = read_clean_dataset()
dif = []
for h, c in zip(d.articleHeadline, d.claimHeadline):
    dif.append(len(h) - len(c))
difference = pd.DataFrame()
difference['dif'] = dif

difference.to_pickle(PICKLED_FEATURES_PATH+"length_diff.pkl")





