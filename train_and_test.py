from Model import Model


trainingSettingsLogisticRegression = {
    "penalty": "l1",  # can be 'l1', 'l2', 'elasticnet', 'none'
    "solver": 'liblinear',
    "max_iter": 1000,  # 100 is the default value
    "n_jobs": -1,  # The number of cores to use, -1 means all
    "random_state": 0,  # The seed for the random number generator used to shuffle the data
    "cross_val_folds": 10

}

trainingSettingsNaiveBayes = {
    "cross_val_folds": 10,
    "verbose": 1
}

trainingSettingsSVM = {
    "cross_val_folds": 10,
    "kernel": 'linear',
    "gamma": 'scale'
}


trainingSettingsRandomForest = {
    "cross_val_folds": 10,
    "max_depth": 6,
    "random_state": 0
}


def forwardSelection():
    all_features = ["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "SVO_ppdb", "word2vec"]
    best_features = []

    for i in range(5):
        # print(best_features)
        max_accuracy_temp = 0
        feature_to_add = ""
        for feature in all_features:
            best_features.append(feature)  # Add the feature to test to the list
            print("TESTING: ", best_features)

            # Train the model with the given features
            model = Model(
                "train_and_test",
                features=best_features,
                classifier="SVM",
                settings=trainingSettingsSVM
            )

            if(model.results > max_accuracy_temp):
                max_accuracy_temp = model.results
                feature_to_add = feature

            best_features.remove(feature)

        best_features.append(feature_to_add)
        all_features.remove(feature_to_add)

        print("BEST FEATURES: ", best_features)
        print("Accuracy: ", max_accuracy_temp)

def backwardSelection():
    best_features = ["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "SVO_ppdb", "word2vec"]

    while len(best_features) > 5:
        max_accuracy_temp = 0
        feature_to_remove = ""

        testing_features = best_features.copy()

        for feature_being_removed in best_features:
            testing_features.remove(feature_being_removed)
            model = Model(
                "train_and_test",
                features=best_features,
                classifier="SVM",
                settings=trainingSettingsSVM
            )
            if(model.results > max_accuracy_temp):
                max_accuracy_temp = model.results
                feature_to_remove = feature_being_removed

            testing_features.append(feature_being_removed)

        best_features.remove(feature_to_remove)
        print(best_features)
        print(max_accuracy_temp)

    print(Model(
        "train_and_test",
        features=best_features,
        classifier="Naive Bayes",
        settings=trainingSettingsNaiveBayes
    ).results)


forwardSelection()

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
# randomForestModel = Model(
#     "train_and_test",
#     features=["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"],
#     classifier="Random Forest",
#     settings=trainingSettingsRandomForest
# )

# print("Accuracy: ", logisticRegressionModel.results)
# print("Accuracy: ", naiveBayesModel.results)
# print("Accuracy: ", svmModel.results)
# print("Accuracy: ", randomForestModel.results)

