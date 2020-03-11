from evaluation.model import Model
from evaluation.model_settings import *

def forwardSelection(classifier, trainingSettings):
    all_features = ["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "SVO_ppdb", "word2vec", "root_dist"]
    best_features = []

    for i in range(5):
        # print(best_features)
        max_f1_temp = 0
        feature_to_add = ""
        for feature in all_features:
            best_features.append(feature)  # Add the feature to test to the list
            # print("TESTING: ", best_features)

            # Train the model with the given features
            model = Model(
                "train_and_test",
                features=best_features,
                classifier=classifier,
                settings=trainingSettings
            )

            if(model.results["f1_score"] > max_f1_temp):
                max_f1_temp = model.results["f1_score"]
                feature_to_add = feature

            best_features.remove(feature)

        print('adding', feature_to_add)
        best_features.append(feature_to_add)
        all_features.remove(feature_to_add)

    print("BEST FEATURES: ", best_features)
    print("F1: ", max_f1_temp)


def backwardSelection(classifier, trainingSettings):
    best_features = ["bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "SVO_ppdb", "word2vec", "root_dist"]

    while len(best_features) > 5:
        max_f1_temp = 0
        feature_to_remove = ""

        testing_features = best_features.copy()

        for feature_being_removed in best_features:
            testing_features.remove(feature_being_removed)
            model = Model(
                "train_and_test",
                features=best_features,
                classifier=classifier,
                settings=trainingSettings
            )
            if(model.results["f1_score"] > max_f1_temp):
                max_f1_temp = model.results["f1_score"]
                feature_to_remove = feature_being_removed

            testing_features.append(feature_being_removed)

        print('removing', feature_to_remove)
        best_features.remove(feature_to_remove)
    print(best_features)
    print(max_f1_temp)


#forwardSelection("Logistic Regression", trainingSettingsLogisticRegression)
backwardSelection("Logistic Regression", trainingSettingsLogisticRegression)
