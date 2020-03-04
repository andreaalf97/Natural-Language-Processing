from data_reading.preprocess_data import apply_lower_case, apply_lemmatization
from data_reading.read_data import read_clean_dataset, PICKLED_FEATURES_PATH

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# List of words taken from github repo of the paper
refutingWords = [
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


hedgingWords = [
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


dictionary = refutingWords + hedgingWords


#presenceOnly is a trigger for future tests to check if performance improves
# by analyzing presence of words instead of counts
def extract_reduced_bow(headlines, useStopwords = False, presenceOnly = False):

    sw = stopwords.words('english') if useStopwords else None

    #CountVectorizer converts a collection of text documents to a matrix of token counts
    cv = CountVectorizer(
        input = "content", #The input is expected to be a series of items
        lowercase = True, #Converts all word to lowercase first --> might not be necessary if we lowercase in advance
        stop_words = sw,  #the list of stopwords
        vocabulary = dictionary, #the features
        binary = presenceOnly, #see above
    )

    bow = cv.fit_transform(headlines)

    return cv.get_feature_names(), bow



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

d.to_pickle(PICKLED_FEATURES_PATH+"ref_hedg_bow.pkl")  # pickle the dataframe to the specified folder
