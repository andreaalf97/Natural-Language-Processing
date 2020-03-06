""" Perform hypothesis testing to see whether the features used are discriminative between the
multiple labels"""

from data_reading.read_data import read_pickle_file, read_clean_dataset
from scipy.stats import mannwhitneyu, ttest_ind_from_stats, shapiro
import pandas as pd

_feature_file_map = {
    'BoW': 'bow.pkl',
    'SVO': 'SVO.pkl',
    'Q': 'q_features.pkl',
    'word2vec': 'word2vec.pkl'
}


def length_difference(dataset: pd.DataFrame):
    """ Explore the hypothesis that some of the groups have
    different title distributions when comparing them to the claim """
    dif = []
    for h, c in zip(dataset.articleHeadline, dataset.claimHeadline):
        dif.append(len(h) - len(c))

    # Add column
    dataset['dif'] = dif
    f = dataset[dataset.articleHeadlineStance == 'for']['dif']
    a = dataset[dataset.articleHeadlineStance == 'against']['dif']
    o = dataset[dataset.articleHeadlineStance == 'observing']['dif']

    print(f.describe())
    print(a.describe())
    print(o.describe())

    # Calculate for normality:
    _, pf = shapiro(f)
    _, po = shapiro(o)
    _, pa = shapiro(a)

    # None are normaly distributed
    print(f"""Test for normality (Length):
                   1) For: {pf}
                   2) Observing : {po}
                   3) Against : {pa}""")

    # Calculate p-values
    _, p_fa = mannwhitneyu(f, a)
    _, p_fo = mannwhitneyu(f, o)
    _, p_oa = mannwhitneyu(o, a)

    print(f"""P-values:
            1) For - Against: {p_fa}
            2) Observing - Against: {p_oa}
            3) For - Observing: {p_fo}""")


def q_counts():
    # Statistics : Q-Feature (mean, std, number_samples)
    f = {'ends': (0.00885, 0.09388, 1238),
         'contains': (0.022617, 0.14874, 1238)}
    o = {'ends': (0.090437, 0.286955, 962),
         'contains': (0.133056, 0.339812, 962)}
    a = {'ends': (0.025316, 0.157284, 395),
         'contains': (0.075949, 0.265253, 395)}
    # Run the t-test!
    for feature in ['ends', 'contains']:
        mean_f, std_f, n_f = f[feature]
        mean_a, std_a, n_a = a[feature]
        mean_o, std_o, n_o = o[feature]
        # Run the actual test
        _, p_fo = ttest_ind_from_stats(mean1=mean_f, std1=std_f, nobs1=n_f,
                                       mean2=mean_o, std2=std_o, nobs2=n_o)
        _, p_fa = ttest_ind_from_stats(mean1=mean_f, std1=std_f, nobs1=n_f,
                                       mean2=mean_a, std2=std_a, nobs2=n_a)
        _, p_ao = ttest_ind_from_stats(mean1=mean_a, std1=std_a, nobs1=n_a,
                                       mean2=mean_o, std2=std_o, nobs2=n_o)

        print(f"""P-values ({feature})
                    1) For - Against: {p_fa}
                    2) Observing - Against: {p_ao}
                    3) For - Observing: {p_fo}""")


def word2vec_stats():
    d = read_clean_dataset()
    w2v = read_pickle_file(_feature_file_map['word2vec'])
    # Divide into the several datasets

    d['w2v'] = w2v.avg_similarity

    f = d[d.articleHeadlineStance == 'for']['w2v']
    a = d[d.articleHeadlineStance == 'against']['w2v']
    o = d[d.articleHeadlineStance == 'observing']['w2v']

    # Calculate for normality:
    _, pf = shapiro(f)
    _, po = shapiro(o)
    _, pa = shapiro(a)

    # None are normaly distributed
    print(f"""Test for normality (W2V):
                1) For: {pf}
                2) Observing : {po}
                3) Against : {pa}""")

    # Calculate p-values
    _, p_fa = mannwhitneyu(f, a)
    _, p_fo = mannwhitneyu(f, o)
    _, p_oa = mannwhitneyu(o, a)

    print(f"""P-values (W2V):
            1) For - Against: {p_fa}
            2) Observing - Against: {p_oa}
            3) For - Observing: {p_fo}""")


if __name__ == '__main__':
    d = read_clean_dataset()
    length_difference(d)
    q_counts()
    word2vec_stats()
