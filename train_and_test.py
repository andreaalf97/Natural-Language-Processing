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
    "cross_val_folds": 10
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

logisticRegressionModel = Model(
    "train_and_test",
    features=['root_dist', 'length_diff', 'q_features'],
    classifier="Logistic Regression",
    settings=trainingSettingsLogisticRegression
)

naiveBayesModel = Model(
    "train_and_test",
    features=['root_dist', 'length_diff', 'q_features'],
    classifier="Naive Bayes",
    settings=trainingSettingsNaiveBayes
)

print("Accuracy LR: ", logisticRegressionModel.results)
# print('Coefficients Logistic Regression: ', np.mean(logisticRegressionModel.model.coef_, axis=0))

print("Accuracy NB: ", naiveBayesModel.results)

svmModel = Model(
    "train_and_test",
    features=['root_dist', 'length_diff', 'q_features'],
    classifier="SVM",
    settings=trainingSettingsSVM
)

randomForestModel = Model(
    "train_and_test",
    features=['root_dist', 'length_diff', 'q_features'],
    classifier="Random Forest",
    settings=trainingSettingsRandomForest
)

print("Accuracy: ", svmModel.results)
print("Accuracy: ", randomForestModel.results)
