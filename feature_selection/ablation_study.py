from Model import Model
from data_reading.read_data import DATA_PATH
from train_and_test import trainingSettingsLogisticRegression, trainingSettingsNaiveBayes, trainingSettingsSVM
import json

feature_set = {"bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO", "word2vec"}
classifiers = [("Logistic Regression", trainingSettingsLogisticRegression), ("Naive Bayes", trainingSettingsNaiveBayes), ("SVM", trainingSettingsSVM)]
results = {}
for feature in feature_set:
    new_feature_set = set(feature_set).remove(feature)
    results[feature] = {}
    for (clsf, settings) in classifiers:
        model = Model(
            "train_and_test",
            features=new_feature_set,
            classifier=clsf,
            settings=settings
        )
        results[feature][clsf] = model.results

with open(DATA_PATH + "/ablation_results.json", "w") as fp:
    json.dump(results, fp,  indent=4)
