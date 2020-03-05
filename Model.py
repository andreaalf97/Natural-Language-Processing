import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from data_reading.read_data import PICKLED_FEATURES_PATH


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
    def __init__(self, index=0, features=[], classifier="", classifierHyperparams=[], trainingSettings=[], test=""):
        self.id = index
        self.features = features
        self.test = test
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
            file = open(PICKLED_FEATURES_PATH + feature_name + ".pkl", "rb")  # Open the pickle file containing
            df = pickle.load(file)  # transforms the pickle file in a pandas DataFrame

            finalDF = pd.concat([finalDF, df], axis=1)  # Adds the new columns to the final dataframe

            file.close()  # Close the file

        return finalDF

    # Applies the selected classifier with any hyper parameters specified
    def trainOnData(self):
        if self.classifier is "Naive Bayes":
            return self.naiveBayes()
        elif self.classifier is "Logistic Regression":
            return self.logisticRegression()
        elif self.classifier is "SVM":
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
        return None

    # Implementation of Cross Validation
    def crossValidation(self):
        return None
