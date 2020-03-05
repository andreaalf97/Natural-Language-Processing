import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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
        return None

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
