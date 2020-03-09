from evaluation.model import Model
from evaluation.model_settings import *

def evaluateSVM(settings, grid={}):
    svmModel = Model(
        "train_and_test",
        features=["root_dist", "q_features", "length_diff"],
        classifier="SVM",
        settings=settings,
        hyperparameters_grid=grid
    )
    print("Results SVM: ", svmModel.results)


def evaluateLogisticRegression(settings):
    logisticRegressionModel = Model(
        "train_and_test",
        features=["root_dist", "q_features", "length_diff"],
        classifier="Logistic Regression",
        settings=settings
    )
    print("Results Logistic Regression: ", logisticRegressionModel.results)


def evaluateNaiveBayes(settings):
    naiveBayesModel = Model(
        "train_and_test",
        features=["root_dist", "q_features", "length_diff"],
        classifier="Naive Bayes",
        settings=settings
    )
    print("Results Naïve Bayes: ", naiveBayesModel.results)


def evaluateRandomForest(settings, grid={}):
    randomForestModel = Model(
        "train_and_test",
        features=["root_dist", "q_features", "length_diff"],
        classifier="Random Forest",
        settings=settings,
        hyperparameters_grid=grid
    )
    print("Results Random Forest: ", randomForestModel.results)


evaluateRandomForest(nestedRandomForestSettings, nestedRandomForestGrid)
