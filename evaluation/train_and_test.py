from evaluation.model import Model
from evaluation.model_settings import *


def evaluateSVM(settings, grid={}):
    svmModel = Model(
        "train_and_test",
        features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb",
                  "word2vec"],
        classifier="SVM",
        settings=settings,
        hyperparameters_grid=grid
    )
    print("Results SVM: ", svmModel.results)
    return svmModel


def evaluateLogisticRegression(settings):
    logisticRegressionModel = Model(
        "train_and_test",
        features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb",
                  "word2vec"],
        classifier="Logistic Regression",
        settings=settings
    )
    print("Results Logistic Regression: ", logisticRegressionModel.results)
    return logisticRegressionModel


def evaluateNaiveBayes(settings):
    naiveBayesModel = Model(
        "train_and_test",
        features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb",
                  "word2vec"],
        classifier="Naive Bayes",
        settings=settings
    )
    print("Results Naïve Bayes: ", naiveBayesModel.results)
    return naiveBayesModel


def evaluateRandomForest(settings, grid={}):
    randomForestModel = Model(
        "train_and_test",
        features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb",
                  "word2vec"],
        classifier="Random Forest",
        settings=settings,
        hyperparameters_grid=grid
    )
    print("Results Random Forest: ", randomForestModel.results)
    return randomForestModel


if __name__ == '__main__':
    #lr = evaluateLogisticRegression(trainingSettingsLogisticRegression)
    #nb = evaluateNaiveBayes(trainingSettingsNaiveBayes)
    #rf = evaluateRandomForest(nestedRandomForestSettings, nestedRandomForestGrid)
    sv = evaluateSVM(nestedSVMSettings, nestedSVMGrid)
