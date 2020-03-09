from Model import Model
import numpy as np

trainingSettingsLogisticRegression = {
    "penalty": "l1",  # can be 'l1', 'l2', 'elasticnet', 'none'
    "solver": 'liblinear',
    "max_iter": 150,  # 100 is the default value
    "n_jobs": -1,  # The number of cores to use, -1 means all
    "random_state": 0,  # The seed for the random number generator used to shuffle the data
    "cross_val_folds": 10
}

trainingSettingsNaiveBayes = {
    "cross_val_folds": 10,
    "verbose": 1
}

trainingSettingsSVM = {
    "C": 2,
    'random_state': 0,
    "cross_val_folds": 10,
    "kernel": 'poly',
    "degree": 2,
    "gamma": 'scale'
}


trainingSettingsRandomForest = {
    "cross_val_folds": 10,
    "max_depth": 5,
    "n_estimators": 25,
    "random_state": 0
}

nestedRandomForestSettings = {
    "outer_cross_val_folds": 5,
    "inner_cross_val_folds": 5,
}

nestedSVMSettings = {
    "outer_cross_val_folds": 5,
    "inner_cross_val_folds": 5,
}

nestedRandomForestGrid = {
    "max_depth": [1, 2, 5, 10, 20, 50],
    "n_estimators": [10, 15, 25, 50, 75, 100],
    "random_state": [0],
}

nestedSVMGrid = {
    "kernel": ['poly'],
    "C": [0.01, 0.1,0.5, 1, 1.5, 2],
    'degree': [2, 3],
    "random_state": [0],
}

svmModel = Model(
    "train_and_test",
    features=["root_dist", "q_features", "length_diff"],
    classifier="SVM",
    settings=nestedSVMSettings,
    hyperparameters_grid=nestedSVMGrid

)

logisticRegressionModel = Model(
    "train_and_test",
    features=["root_dist", "q_features", "length_diff"],
    classifier="Logistic Regression",
    settings=trainingSettingsLogisticRegression
)
#
naiveBayesModel = Model(
    "train_and_test",
    features=["root_dist", "q_features", "length_diff"],
    classifier="Naive Bayes",
    settings=trainingSettingsNaiveBayes
)



randomForestModel = Model(
    "train_and_test",
    features=["root_dist", "q_features", "length_diff"],
    classifier="Random Forest",
    settings=nestedRandomForestSettings,
    hyperparameters_grid=nestedRandomForestGrid
)

print("Results Logistic Regression: ", logisticRegressionModel.results)
print("Results Naïve Bayes: ", naiveBayesModel.results)
print("Results SVM: ", svmModel.results)
print("Results Random Forest: ", randomForestModel.results)

