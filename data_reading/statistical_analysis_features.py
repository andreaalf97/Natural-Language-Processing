""" Perform hypothesis testing to see whether the features used are discriminative between the
multiple labels"""

from data_reading.read_data import read_pickle_file, read_clean_dataset
from scipy.stats import mannwhitneyu, ttest_ind_from_stats, shapiro
import pandas as pd
from scipy import stats

# Shapiro-Wilk test is used to test wheter feature has normal distribution

# Mann-Whitney U test can be used to investigate whether two independent samples were selected from populations having the same distribution.

# Chi-sqaured test is used to test dependence of 2 categorical features

_feature_file_map = {
    'BoW': 'bow.pkl',
    'SVO': 'SVO.pkl',
    'Q': 'q_features.pkl',
    'word2vec': 'word2vec.pkl',
    'root_dist': 'root_dist.pkl',
    'kuhn_munkres': 'kuhn_munkres.pkl',
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
    f = {'q_ends': (0.00885, 0.09388, 1238),
         'q_contains': (0.022617, 0.14874, 1238)}
    o = {'q_ends': (0.090437, 0.286955, 962),
         'q_contains': (0.133056, 0.339812, 962)}
    a = {'q_ends': (0.025316, 0.157284, 395),
         'q_contains': (0.075949, 0.265253, 395)}

    d = read_clean_dataset()
    q = read_pickle_file(_feature_file_map['Q'])
    q['Stance'] = d.articleHeadlineStance

    # Run the t-test!
    for feature in ['q_ends', 'q_contains']:
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

        # Chi-square test for dependency between feature and stance
        contingency_table = pd.crosstab(q['Stance'],
                                        q[feature],
                                        margins=False)

        chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

        print("\n")
        print(f"""=== Chi2 Stat ({feature}) ===""")
        print(chi2_stat)
        print("\n")
        print("===Degrees of Freedom===")
        print(dof)
        print("\n")
        print("===P-Value===")
        print(p_val)
        print("\n")
        print("===Contingency Table===")
        print(ex)


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


def root_dist_stats():
    d = read_clean_dataset()
    km = read_pickle_file(_feature_file_map['root_dist'])
    # Divide into the several datasets

    for feature in ['refute_dist', 'hedge_dist']:

        d[feature] = km[feature]
        f = d[d.articleHeadlineStance == 'for'][feature]
        a = d[d.articleHeadlineStance == 'against'][feature]
        o = d[d.articleHeadlineStance == 'observing'][feature]

        # Calculate for normality:
        _, pf_r = shapiro(f)
        _, po_r = shapiro(o)
        _, pa_r = shapiro(a)

        print(f"""Test for normality ({feature}):
                        1) For: {pf_r}
                        2) Observing : {po_r}
                        3) Against : {pa_r}""")

        _, p_fa = mannwhitneyu(f, a)
        _, p_fo = mannwhitneyu(f, o)
        _, p_oa = mannwhitneyu(o, a)

        print(f"""P-values ({feature}):
                    1) For - Against: {p_fa}
                    2) Observing - Against: {p_oa}
                    3) For - Observing: {p_fo}""")


def kuhn_munkres_stats():
    d = read_clean_dataset()
    km = read_pickle_file(_feature_file_map['kuhn_munkres'])
    # Divide into the several datasets

    d['kuhn_munkres'] = km['Kuhn-Munkres']

    f = d[d.articleHeadlineStance == 'for']['kuhn_munkres']
    a = d[d.articleHeadlineStance == 'against']['kuhn_munkres']
    o = d[d.articleHeadlineStance == 'observing']['kuhn_munkres']

    # Calculate for normality:
    _, pf = shapiro(f)
    _, po = shapiro(o)
    _, pa = shapiro(a)

    # None are normaly distributed
    print(f"""Test for normality (K-M):
                    1) For: {pf}
                    2) Observing : {po}
                    3) Against : {pa}""")

    # Calculate p-values
    _, p_fa = mannwhitneyu(f, a)
    _, p_fo = mannwhitneyu(f, o)
    _, p_oa = mannwhitneyu(o, a)

    print(f"""P-values (K-M):
                1) For - Against: {p_fa}
                2) Observing - Against: {p_oa}
                3) For - Observing: {p_fo}""")


if __name__ == '__main__':
    root_dist_stats()
