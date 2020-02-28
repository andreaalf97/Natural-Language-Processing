import pandas as pd
from collections import Counter

file = "../data/url-versions-2015-06-14.csv"
clean_file = "../data/url-versions-2015-06-14-clean.csv"


def load_dataset():
    d = pd.read_csv(file)
    cd = pd.read_csv(clean_file)
    return d, cd


# Load datasets
data, clean_data = load_dataset()
print('Columns in one not in the other', set(data.columns) - set(clean_data.columns))

# Keep the appropriate columns
df = data[clean_data.columns[1:]].dropna().drop_duplicates(subset='articleId')

# Drop ignores
df = df[df.articleHeadlineStance!='ignoring']

# Show the distribution of the labels in the data
labels_clean = Counter(clean_data.articleHeadlineStance)
labels = Counter(df.articleHeadlineStance)

print("Clean dataset labels:", labels_clean.items())
print("Labels:", labels.items())

# Until now I did not find a way of exactly get the same columns as he did
# As you can see, there are less for and against, but more observing
# Plus, there are some articles that appear as 'ignoring' in the original dataset and are still added to the clean one
ig = data[data.articleHeadlineStance=='ignoring']['articleId']
weird_stuff = [c for c in ig.to_list() if c in clean_data.articleId.to_list()]

# All these are articles with no stance (ignoring) but then they appear magically with another stance
print('Amount of "ignoring" articles included in the final dataset:', len(weird_stuff))

# And the final lengths of the dataset differ slightly
print(f"Length of the dataset used by him: {len(clean_data)}\nLenght of our dataset: {len(df)}")

# Possible thing he did to combine the article headline stance and the article stance
# d['stance'] = np.where(d.articleHeadlineStance!='ignoring', d.articleHeadlineStance, d.articleStance)
