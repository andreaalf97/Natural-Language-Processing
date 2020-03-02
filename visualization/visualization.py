import matplotlib.pyplot as plt
import pandas as pd


# Methods

# Plots a histogram showing stances within the data provided
def stance_histogram(stances, title=""):
    plt.hist(stances, bins=range(4), edgecolor='white', linewidth=1)
    plt.title(title)
    plt.show()


# Plots a bar chart containing top 5 words in the bow list provided
def bow_chart(bow, title="", items=5):
    data = bow.head(items)
    xpos = 0
    ticks = []
    for row in data.iterrows():
        plt.bar(xpos, row[1])
        ticks.append(row[0])
        # Ensure enough spacing between bars
        xpos = xpos + 1
    plt.xticks(range(items), ticks)
    plt.title(title)
    plt.show()

#Plots a bar chart showing how the headlines are distributed by q value
def plot_qData(qData, title=""):
    contains = 0
    ends = 0
    does_not_contain = 0
    for row in qData.iterrows():
        tuple = row[1]
        # Does not contain
        if not tuple[0] and not tuple[1]:
            does_not_contain = does_not_contain + 1
        elif tuple[0] and not tuple[1]:
            ends = ends + 1
        elif not tuple[0] and tuple[1]:
            contains = contains + 1
        else:
            ends = ends + 1
    plt.bar(0, contains)
    plt.bar(1, ends)
    plt.bar(2, does_not_contain)
    ticks = ["Contains", "Ends", "Does not Contain"]
    plt.xticks(range(4), ticks)
    plt.title(title)
    plt.show()

def root_dist_hist(data):
    refute_dist = data["refute_dist"]
    hedge_dist = data["hedge_dist"]
    plt.hist(refute_dist)
    plt.title("Refute Distance")
    plt.show()
    plt.hist(hedge_dist)
    plt.title("Hedge Distance")
    plt.show()

# Main Visualization Logic
data = pd.read_csv("../data/url-versions-2015-06-14-clean.csv")
stance_histogram(data["articleHeadlineStance"], "Clean Data Stance Frequency")

bow = pd.read_pickle("../data/pickled_features/bow.pkl")
# Add up the values
total = bow.sum(axis=0, skipna=True).to_frame()
total = total.sort_values(by=0, ascending=False)
# Display 20 most common
bow_chart(total, "Top 5 Tokens")
# Display all
# bow_chart(total, "all", len(total))
# Display 20 least common
total = total.sort_values(by=0, ascending=True)
bow_chart(total, "Lowest 5 Tokens")

qData = pd.read_pickle("../data/pickled_features/q_features.pkl")
plot_qData(qData, "Distribution of Headlines by Q")

rootData = pd.read_pickle("../data/pickled_features/root_dist.pkl")
rootData = rootData.replace(100000, 0)
root_dist_hist(rootData)

vecData = pd.read_pickle("../data/pickled_features/word2vec.pkl")