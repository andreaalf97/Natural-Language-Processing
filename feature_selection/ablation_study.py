from evaluation.model import Model
from data_reading.read_data import DATA_PATH
from evaluation.train_and_test import trainingSettingsLogisticRegression, trainingSettingsNaiveBayes, \
    trainingSettingsSVM, trainingSettingsRandomForest, nestedRandomForestGrid, nestedRandomForestSettings, nestedSVMSettings
import json

feature_set = {"bow", "kuhn_munkres", "length_diff", "q_features", "ref_hedg_bow", "root_dist", "SVO_ppdb", "word2vec"}
classifiers = [("Logistic Regression", trainingSettingsLogisticRegression, {}),
               ("Naive Bayes", trainingSettingsNaiveBayes, {}),
               ("Random Forest", nestedRandomForestSettings, nestedRandomForestGrid),
               ("SVM", trainingSettingsSVM, nestedSVMSettings)]
results = {}
for feature in feature_set:
    new_feature_set = feature_set.copy()
    new_feature_set.remove(feature)
    results[feature] = {}
    for (clsf, settings, hyper) in classifiers:
        print(f'Fitting model {clsf} with parameters {settings} and hyper {hyper}')
        model = Model(
            "train_and_test",
            features=new_feature_set,
            classifier=clsf,
            settings=settings,
            hyperparameters_grid=hyper
        )
        results[feature][clsf] = model.results

with open(DATA_PATH + "/ablation_results.json", "w") as fp:
    json.dump(results, fp, indent=4)
