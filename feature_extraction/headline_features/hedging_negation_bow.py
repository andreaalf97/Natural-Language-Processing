from data_reading.preprocess_data import apply_lower_case, apply_lemmatization
from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH

import pandas as pd
import sklearn



def extract_reduced_bow(headlines):
    return 0, 0



# Lemmatize the dataset for better representation
dataset = read_clean_dataset()  # Read the dataset
dataset = apply_lower_case(dataset)
dataset = apply_lemmatization(dataset)



features, b = extract_reduced_bow(dataset.articleHeadline.values)

# create a dataset
d = pd.DataFrame(b.toarray())
d.columns = features

# This is the old code to make use of the old bow representation with no optimizations or tf-idf
# counts = create_corpus(dataset)  # Number of occurrences of each word in the corpus
# assignments = dict(zip(counts.keys(), range(len(counts))))  # Index of each of the words in the vector
# print(counts)
# d = create_vectors(dataset, assignments)  # dataframe with all the vectors
#

d.to_pickle(PICKLED_FEATURES_PATH+"bow.pkl")  # pickle the dataframe to the specified folder
