import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from data_reading.read_data import read_pickle_file, read_clean_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


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
    def __init__(self, index=0, features=[], classifier="", settings={}, test="", hyperparameters_grid={}):
        self.id = index
        self.features = features
        self.test = test
        self.classifier = classifier
        self.trainingSettings = settings
        self.model = None
        self.hyperparameters_grid = hyperparameters_grid
        self.labels = read_clean_dataset()['articleHeadlineStance']
        self.featureMatrix = self.constructFeaturesMatrix()
        # Compute train and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.featureMatrix, self.labels,
                                                                                test_size=0.2, random_state=0,
                                                                                stratify=self.labels)
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

        return finalDF

    def calc_confusion_matrix(self):
        predictions = cross_val_predict(self.model, self.featureMatrix, self.labels,
                                        cv=self.trainingSettings["cross_val_folds"], verbose=1)
        self.confusion_matrix = confusion_matrix(self.labels, predictions, labels=["for", "observing", "against"])

    def calcConfusionMatrixScore(self):
        trace = np.trace(self.confusion_matrix)
        total = np.sum(self.confusion_matrix)
        return trace / total

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
                                          'roc_auc_ovr_weighted'], n_jobs=-1)
        self.model = self.model.fit(self.X_train, self.y_train)
        # Compute F1 score to return
        y_pred = self.model.predict(self.X_test)
        f_score = f1_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        #auc = roc_auc_score(self.y_test, y_pred, average='weighted', multi_class='ovr')
        return {'f1_score': f_score,
                'precision': precision,
                'recall': recall,
                'cross_val_results': {k: (np.mean(v), np.std(v)) for k, v in
                                      results.items()}}

    # Implementation of svm
    def SVM(self):
        if self.hyperparameters_grid:
            self.model = svm.SVC(probability=True)

            # To be used within GridSearch
            inner_cv = StratifiedKFold(n_splits=self.trainingSettings["inner_cross_val_folds"], shuffle=True,
                                       random_state=0)

            # To be used in outer CV
            outer_cv = StratifiedKFold(n_splits=self.trainingSettings["outer_cross_val_folds"], shuffle=True,
                                       random_state=0)

            clf = GridSearchCV(estimator=self.model, param_grid=self.hyperparameters_grid, cv=inner_cv, verbose=1, n_jobs=-1, scoring='f1_weighted')

            clf.fit(self.X_train, self.y_train)
            print(clf.best_estimator_)
            self.model = clf.best_estimator_

            nested_score = cross_validate(clf.best_estimator_, X=self.X_train, y=self.y_train, cv=outer_cv,
                                          scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc_ovr_weighted'])
            # Compute F1 score to return
            y_pred = self.model.predict(self.X_test)
            f_score = f1_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
            # auc = roc_auc_score(self.y_test, y_pred, average='weighted', multi_class='ovr')
            return {'f1_score': f_score,
                    'precision': precision,
                    'recall': recall,
                    'cross_val_results': {k: (np.mean(v), np.std(v)) for k, v in
                                          nested_score.items()}}

        else:
            self.model = svm.SVC(C=self.trainingSettings['C'], gamma=self.trainingSettings["gamma"],
                                 kernel=self.trainingSettings["kernel"],
                                 random_state=self.trainingSettings['random_state'],
                                 degree=self.trainingSettings['degree'], probability=True)
            results = cross_validate(self.model, self.featureMatrix, self.labels,
                                     cv=self.trainingSettings["cross_val_folds"], verbose=1,
                                     scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                              'roc_auc_ovr_weighted'], n_jobs=-1)

            results = {k: np.mean(v) for k, v in results.items()}
            return results

    # Implementation of logistic regression
    def logisticRegression(self):
        self.model = LogisticRegression(
            penalty=self.trainingSettings["penalty"],
            max_iter=self.trainingSettings["max_iter"],
            n_jobs=self.trainingSettings["n_jobs"],
            random_state=self.trainingSettings["random_state"],
            solver=self.trainingSettings['solver']
        )

        results = \
            cross_validate(self.model, self.X_train, self.y_train, cv=self.trainingSettings["cross_val_folds"],
                           verbose=1,
                           scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                    'roc_auc_ovr_weighted'], n_jobs=-1)

        print(results)
        self.model = self.model.fit(self.X_train, self.y_train)

        # Compute F1 score to return
        y_pred = self.model.predict(self.X_test)
        f_score = f1_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        #auc = roc_auc_score(self.y_test, y_pred, average='weighted', multi_class='ovr')
        return {'f1_score': f_score,
                'precision': precision,
                'recall': recall,
                'cross_val_results': {k: (np.mean(v), np.std(v)) for k, v in
                                      results.items()}}

    # Implementation of randomForest
    def randomForest(self):
        if self.hyperparameters_grid:
            self.model = RandomForestClassifier()

            # To be used within GridSearch
            inner_cv = StratifiedKFold(n_splits=self.trainingSettings["inner_cross_val_folds"], shuffle=True,
                                       random_state=0)

            # To be used in outer CV
            outer_cv = StratifiedKFold(n_splits=self.trainingSettings["outer_cross_val_folds"], shuffle=True,
                                       random_state=0)

            clf = GridSearchCV(estimator=self.model, param_grid=self.hyperparameters_grid, cv=inner_cv, n_jobs=-1,
                               verbose=1, scoring='f1_weighted')

            clf.fit(self.X_train, self.y_train)
            print(clf.best_estimator_)
            self.model = clf.best_estimator_

            nested_score = cross_validate(clf.best_estimator_, X=self.featureMatrix, y=self.labels, cv=outer_cv,
                                          scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                                   'roc_auc_ovr_weighted'], n_jobs=-1, verbose=1)
            # Compute F1 score to return
            # TODO calculate more than f_score, precision, recall and auc
            y_pred = self.model.predict(self.X_test)
            f_score = f1_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
           # auc = roc_auc_score(self.y_test, y_pred, average='weighted', multi_class='ovr')
            return {'f1_score': f_score,
                    'precision': precision,
                    'recall': recall,
                    'cross_val_results': {k: (np.mean(v), np.std(v)) for k, v in
                                          nested_score.items()}}

        else:
            # Initialize the model
            self.model = RandomForestClassifier(
                n_estimators=self.trainingSettings['n_estimators'],
                max_depth=self.trainingSettings["max_depth"],
                random_state=self.trainingSettings["random_state"]
            )
            results = cross_validate(self.model, self.featureMatrix, self.labels,
                                     cv=self.trainingSettings["cross_val_folds"], verbose=1,
                                     scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                              'roc_auc_ovr_weighted'], n_jobs=-1)
            results = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
            return results
