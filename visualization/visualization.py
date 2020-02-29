import matplotlib.pyplot as plt
import pandas as pd

# Plots a histogram showing stances within the data provided
def stance_histogram(stances, title=""):
    plt.hist(stances, bins=range(4), edgecolor='white', linewidth=1)
    plt.title(title)
    plt.show()


data = pd.read_pickle("../data/dummy.pkl")
stance_histogram(data["articleHeadlineStance"], "Clean Data Stance Frequency")
