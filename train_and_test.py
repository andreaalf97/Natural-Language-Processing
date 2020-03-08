from Model import Model

trainingSettingsLogisticRegression = {
    "penalty": "l2",  # can be 'l1', 'l2', 'elasticnet', 'none'
    "max_iter": 250,  # 100 is the default value
    "n_jobs": -1,  # The number of cores to use, -1 means all
    "random_state": 0,  # The seed for the random number generator used to shuffle the data
    "cross_val_folds": 10
}

trainingSettingsNaiveBayes = {
    "cross_val_folds": 10
}

trainingSettingsSVM = {
    "cross_val_folds": 10,
    "kernel": 'rbf',
    "gamma": 'scale'
}

trainingSettingsRandomForest = {
    "cross_val_folds": 10,
    "max_depth": 2,
    "random_state": 0
}

logisticRegressionModel = Model(
    "train_and_test",
    features=["bow", "kuhn_munkres"],
    classifier="Logistic Regression",
    settings=trainingSettingsLogisticRegression
)

naiveBayesModel = Model(
    "train_and_test",
    features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
    classifier="Naive Bayes",
    settings=trainingSettingsNaiveBayes
)

svmModel = Model(
    "train_and_test",
    features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
    classifier="SVM",
    settings=trainingSettingsSVM
)

randomForestModel = Model(
    "train_and_test",
    features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
    classifier="Random Forest",
    settings=trainingSettingsRandomForest
)

print("Accuracy: ", logisticRegressionModel.results)
print("Confusion Matrix: ", logisticRegressionModel.confusion_matrix)
print("Accuracy: ", naiveBayesModel.results)
print("Confusion Matrix: ", naiveBayesModel.confusion_matrix)
print("Accuracy: ", svmModel.results)
print("Confusion Matrix: ", svmModel.confusion_matrix)
print("Accuracy: ", randomForestModel.results)
print("Confusion Matrix: ", randomForestModel.confusion_matrix)