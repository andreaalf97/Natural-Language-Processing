import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from data_reading.read_data import read_pickle_file, read_clean_dataset
from sklearn.ensemble import RandomForestClassifier


class Model:
    id = 0
    features = []
    featureMatrix = []
    classifier = ""
    test = ""
    results = 0
    confusion_matrix = []

    # Features is an array of features to use.
    # Classifier is the classifier to use. E.g., SVM
    # settings are any settings for the training and testing such as size of batches or folds in cross validation
    # test is the type of test to use like cross validation
    def __init__(self, index=0, features=[], classifier="", settings={}, test=""):
        self.id = index
        self.features = features
        self.test = test
        self.classifier = classifier
        self.trainingSettings = settings
        self.model = None
        self.labels = read_clean_dataset()['articleHeadlineStance']
        self.featureMatrix = self.constructFeaturesMatrix()
        self.results = self.trainOnData()

    # Used to retrieve features from the appropriate pickle file and construct a matrix
    def constructFeaturesMatrix(self):
        # TODO I'm assuming the features will be given as the name of the file that contains them
        # The features to use are bow, kuhn_munkres, length_diff, q_features (this is still T/F), ref_hedg_bow,
        # root_dist (maybe change from 10000 to 0), SVO, word2vec (many NaN in prod_similarity)

        finalDF = pd.DataFrame()
        for feature_name in self.features:  # TODO load from a file, not Model.features
            df = read_pickle_file(feature_name)  # transforms the pickle file in a pandas DataFrame

            if (feature_name == 'word2vec'):
                df = df['avg_similarity']

            finalDF = pd.concat([finalDF, df], axis=1)  # Adds the new columns to the final dataframe
            print(finalDF.shape)

        return finalDF

    def calc_confusion_matrix(self):
        predictions = cross_val_predict(self.model, self.featureMatrix, self.labels,
                                        cv=self.trainingSettings["cross_val_folds"], verbose=1)
        self.confusion_matrix = confusion_matrix(self.labels, predictions, labels=["for", "observing", "against"])

    # Applies the selected classifier with any hyper parameters specified
    def trainOnData(self):
        if self.classifier == "Naive Bayes":
            return self.naiveBayes()
        elif self.classifier == "Logistic Regression":
            return self.logisticRegression()
        elif self.classifier == "SVM":
            return self.SVM()
        elif self.classifier == "Random Forest":
            return self.randomForest()
        else:
            print("No Classifier Selected")
            return None

    # Implementation of Naive Bayes
    def naiveBayes(self):
        self.model = GaussianNB()

        results = cross_validate(self.model, self.featureMatrix, self.labels,
                                 cv=self.trainingSettings["cross_val_folds"], verbose=1,
                                 scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                          'roc_auc_ovr_weighted'])
        results = {k: np.mean(v) for k, v in results.items()}

        return results

    # Implementation of svm
    def SVM(self):
        self.model = svm.SVC(gamma=self.trainingSettings["gamma"], kernel=self.trainingSettings["kernel"])

        results = cross_validate(self.model, self.featureMatrix, self.labels,
                                 cv=self.trainingSettings["cross_val_folds"], verbose=1,
                                 scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                          'roc_auc_ovr_weighted'])
        results = {k: np.mean(v) for k, v in results.items()}

        return results

    # Implementation of logistic regression
    def logisticRegression(self):
        lrModel = LogisticRegression(
            penalty=self.trainingSettings["penalty"],
            max_iter=self.trainingSettings["max_iter"],
            n_jobs=self.trainingSettings["n_jobs"],
            random_state=self.trainingSettings["random_state"],
            solver=self.trainingSettings['solver']
        )

        results = \
            cross_validate(lrModel, self.featureMatrix, self.labels, cv=self.trainingSettings["cross_val_folds"],
                           verbose=1,
                           scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                    'roc_auc_ovr_weighted'])

        results = {k: np.mean(v) for k, v in results.items()}
        return results

    # Implementation of randomForest
    def randomForest(self):
        # Initialize the model
        self.model = RandomForestClassifier(
            max_depth=self.trainingSettings["max_depth"],
            random_state=self.trainingSettings["random_state"]
        )
        results = cross_validate(self.model, self.featureMatrix, self.labels,
                                 cv=self.trainingSettings["cross_val_folds"], verbose=1,
                                 scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                          'roc_auc_ovr_weighted'])
        results = {k: np.mean(v) for k, v in results.items()}
        return results
