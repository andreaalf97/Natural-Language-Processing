import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from data_reading.read_data import read_pickle_file, read_clean_dataset


class Model:
    id = 0
    features = []
    featureMatrix = []
    classifier = ""
    test = ""
    results = ""

    # Features is an array of features to use.
    # Classifier is the classifier to use. E.g., SVM
    # classifierHyperparams are hyperparameters that are provided when a classifier needs hyperparameters
    # trainingSettings are any settings for the training such as size of batches
    # test is the type of test to use like cross validation
    def __init__(self, index=0, features=[], classifier="", classifierHyperparams=[], trainingSettings={}, test=""):
        self.id = index
        self.features = features
        self.test = test
        self.classifier = classifier
        self.trainingSettings = trainingSettings
        self.labels = read_clean_dataset()['articleHeadlineStance']
        self.featureMatrix = self.constructFeaturesMatrix()
        model = self.trainOnData()
        results = self.testModel()

    # Used to retrieve features from the appropriate pickle file and construct a matrix
    def constructFeaturesMatrix(self):
        # TODO I'm assuming the features will be given as the name of the file that contains them
        # The features to use are bow, kuhn_munkres, length_diff, q_features (this is still T/F), ref_hedg_bow,
        # root_dist (maybe change from 10000 to 0), SVO, word2vec (many NaN in prod_similarity)

        finalDF = pd.DataFrame()
        for feature_name in self.features:  # TODO load from a file, not Model.features
            df = read_pickle_file(feature_name)  # transforms the pickle file in a pandas DataFrame

            finalDF = pd.concat([finalDF, df], axis=1)  # Adds the new columns to the final dataframe

        return finalDF

    # Applies the selected classifier with any hyper parameters specified
    def trainOnData(self):
        if self.classifier == "Naive Bayes":
            return self.naiveBayes()
        elif self.classifier == "Logistic Regression":
            return self.logisticRegression()
        elif self.classifier == "SVM":
            return self.SVM()
        else:
            print("No Classifier Selected")
            return None

    # Test the model using whatever testing method specified
    def testModel(self):
        if self.test is "Cross Validation":
            return self.crossValidation()
        return None

    # Implementation of Naive Bayes
    def naiveBayes(self):
        return None

    # Implementation of svm
    def SVM(self):
        return None

    # Implementation of logistic regression
    def logisticRegression(self):

        # Initialize the model
        lrModel = LogisticRegression(
            penalty = self.trainingSettings["penalty"],
            max_iter = self.trainingSettings["max_iter"],
            n_jobs = self.trainingSettings["n_jobs"],
            random_state = self.trainingSettings["random_state"]
        )

        lrModel.fit(self.featureMatrix, self.labels)

        print("Accuracy on train set: ", lrModel.score(self.featureMatrix, self.labels)*100)

        return lrModel

    # Implementation of Cross Validation
    def crossValidation(self):
        return None
