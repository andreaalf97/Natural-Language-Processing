from Model import Model

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

nestedRandomForestGrid = {
    "max_depth": [1, 2, 5, 10, 20, 50],
    "n_estimators": [10, 15, 25, 50],
    "random_state": [0],
}

# logisticRegressionModel = Model(
#     "train_and_test",
#     features=["bow", "kuhn_munkres"],
#     classifier="Logistic Regression",
#     settings=trainingSettingsLogisticRegression
# )
#
# naiveBayesModel = Model(
#     "train_and_test",
#     features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
#     classifier="Naive Bayes",
#     settings=trainingSettingsNaiveBayes
# )
#
# svmModel = Model(
#     "train_and_test",
#     features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
#     classifier="SVM",
#     settings=trainingSettingsSVM
# )
#
randomForestModel = Model(
    "train_and_test",
    features=["root_dist"],
    classifier="Random Forest",
    settings=nestedRandomForestSettings,
    hyperparameters_grid=nestedRandomForestGrid
)

# print("Results: ", logisticRegressionModel.results)
# print("Results: ", naiveBayesModel.results)
# print("Results: ", svmModel.results)
print("Results: ", randomForestModel.results)

