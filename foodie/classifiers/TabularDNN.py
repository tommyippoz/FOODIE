import numpy
import pandas
from autogluon.tabular import TabularPredictor
from pytorch_tabnet.tab_model import TabNetClassifier

from foodie.classifiers import Classifier
from foodie.utils.general_utils import current_ms


class TabNet(Classifier):
    """
    Wrapper for the torch.tabnet algorithm
    """

    def __init__(self, metric=None, verbose=0):
        Classifier.__init__(self, TabNetClassifier(verbose=verbose))
        self.metric = metric
        self.verbose = verbose

    def fit(self, x_train, y_train):
        if isinstance(x_train, pandas.DataFrame):
            x_train = x_train.to_numpy()
        if self.metric is None:
            self.clf.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024, eval_metric=['auc'],
                         patience=2)
        else:
            self.clf.fit(X_train=x_train, y_train=y_train, max_epochs=40, batch_size=1024,
                         eval_metric=[self.metric], patience=2)
        self.classes_ = numpy.unique(y_train)
        self.feature_importances_ = self.get_feature_importances()

    def get_feature_importances(self):
        return self.clf.feature_importances_

    def predict(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.clf.predict(x_test)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_test = x_test.to_numpy()
        return self.clf.predict_proba(x_test)

    def classifier_name(self):
        return "TabNet"


class AutoGluon(Classifier):
    """
    Wrapper for classifiers taken from Gluon library
    clf_name options are
    ‘GBM’ (LightGBM)
    ‘CAT’ (CatBoost)
    ‘XGB’ (XGBoost)
    ‘RF’ (random forest)
    ‘XT’ (extremely randomized trees)
    ‘KNN’ (k-nearest neighbors)
    ‘LR’ (linear regression)
    ‘NN’ (neural network with MXNet backend)
    ‘FASTAI’ (neural network with FastAI backend)
    """

    def __init__(self, label_name, clf_name, metric, verbose=0):
        Classifier.__init__(self, TabularPredictor(label=label_name, eval_metric=metric, verbosity=verbose))
        self.label_name = label_name
        self.clf_name = clf_name
        self.metric = metric
        self.verbose = verbose
        self.feature_importance = []
        self.feature_names = None

    def fit(self, x_train, y_train):
        path = './AutogluonModels/' + str(current_ms())
        self.classes_ = numpy.unique(y_train)
        self.clf = TabularPredictor(label=self.label_name, eval_metric=self.metric,
                                    path=path, verbosity=self.verbose)
        if self.feature_names is None:
            self.feature_names = ['col' + str(i) for i in range(0, x_train.shape[1])]
        df = pandas.DataFrame(data=x_train.copy(), columns=self.feature_names)
        df[self.label_name] = y_train
        self.clf.fit(train_data=df, hyperparameters={self.clf_name: {}})
        self.feature_importances_ = self.clf.feature_importance(df)

        self.trained = True

    def get_feature_importances(self):
        return self.feature_importances_

    def predict(self, x_test):
        df = pandas.DataFrame(data=x_test, columns=self.feature_names)
        return self.clf.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        df = pandas.DataFrame(data=x_test, columns=self.feature_names)
        return self.clf.predict_proba(df, as_pandas=False)

    def classifier_name(self):
        return "AutoGluon"


class FastAI(AutoGluon):
    """
    Wrapper for the gluon.FastAI algorithm
    """

    def __init__(self, label_name, metric, verbose=0):
        AutoGluon.__init__(self, label_name, "FASTAI", metric, verbose)

    def classifier_name(self):
        return "FastAI"
