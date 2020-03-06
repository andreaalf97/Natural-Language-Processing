from Model import Model

trainingSettingsLogisticRegression = {
    "penalty": "l2",  # can be 'l1', 'l2', 'elasticnet', 'none'
    "max_iter": 100,  # 100 is the default value
    "n_jobs": -1,  # The number of cores to use, -1 means all
    "random_state": 0,  # The seed for the random number generator used to shuffle the data
    "cross_val_folds": 10
}

trainingSettingsNaiveBayes = {
    "cross_val_folds": 10
}

model = Model(
    "train_and_test",
    features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO", "word2vec"],
    classifier="Naive Bayes",
    settings=trainingSettingsNaiveBayes
)


print("Accuracy: ", model.results)