import matplotlib.pyplot as plt
import pandas as pd

# Plots a histogram showing stances within the data provided
def stance_histogram(stances, title=""):
    plt.hist(stances, bins=range(4), edgecolor='white', linewidth=1)
    plt.title(title)
    plt.show()

# Plots a bar chart containing top 5 words in the bow list provided
def bow_chart(bow, title="", items=5):
    data = bow.head(items)
    xpos = 0
    tickPositions = []
    ticks = []
    for row in data.iterrows():
        print(row[1])
        plt.bar(xpos, row[1])
        ticks.append(row[0])
        tickPositions.append(xpos)
        #Ensure enough spacing between bars
        xpos = xpos + 1
    plt.xticks(range(items), ticks)
    plt.title(title)
    plt.show()


data = pd.read_csv("../data/url-versions-2015-06-14-clean.csv")
stance_histogram(data["articleHeadlineStance"], "Clean Data Stance Frequency")

bow = pd.read_pickle("../data/pickled_features/bow.pkl")
# Add up the values
total = bow.sum(axis = 0, skipna = True).to_frame()
total = total.sort_values(by=0, ascending= False)
# Display 20 most common
bow_chart(total, "Top 5 Tokens")
# Display 20 least common
total = total.sort_values(by=0, ascending= True)
bow_chart(total, "Lowest 5 Tokens")

