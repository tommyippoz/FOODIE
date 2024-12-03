import numpy
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array

from foodie.calibration import PostHocCalibrator
from foodie.calibration.PostHocCalibrator import PlattScaling, IsotonicScaling


# ---------------------------------- SUPPORT METHODS ------------------------------------------------------

def get_classifier_name(clf) -> str:
    """
    Reads the object and extracts a name for such a classifier
    :param clf: the classifier object
    :return:
    """
    if isinstance(clf, Classifier):
        if clf.calibrator is None:
            return clf.clf.__class__.__name__ + "#None"
        else:
            return clf.clf.__class__.__name__ + "#" + clf.calibrator.get_name()
    else:
        return clf.__class__.__name__


def get_feature_importance(clf):
    """
    Reads the object and extracts a name for such a classifier
    :param clf: the classifier object
    :return:
    """
    if isinstance(clf, Classifier):
        return clf.feature_importances()
    else:
        return clf.feature_importances_



class Classifier(BaseEstimator, ClassifierMixin):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, clf, calibrator_str: str = None):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        :param calibrator_str: post-hoc calibration strategy to be used in the Classifier
        """
        self.clf = clf
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None
        self.calibrator = self.choose_calibrator(calibrator_str)

    def choose_calibrator(self, calibrator_str: str) -> PostHocCalibrator:
        """
        Function that returns the calibrator object from string
        :param calibrator_str: the string that describes the calibrator (can be None)
        :return: a PostHocCalibrator object, or None
        """
        if calibrator_str in ["platt", "PLATT", "Platt", "plattscaling", "Platt Scaling", "PlattScaling",
                              "PLATTSCALING", "PLATT SCALING"]:
            return PlattScaling(self.clf)
        elif calibrator_str in ["iso", "isotonic", "Iso", "Isotonic", "IsotonicScaling", "Isotonic Scaling",
                                "ISOTONICSCALING", "ISOTONIC SCALING"]:
            return IsotonicScaling(self.clf)
        else:
            return None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        if y is not None:
            X, y = check_X_y(X, y)
            self.classes_ = unique_labels(y)
            if len(self.classes_) < 2:
                print("Classifier '%s' is given with a labelled train set of a single class" % get_classifier_name(self))
                exit(1)
            self.clf.fit(X, y)
            if self.calibrator is not None:
                self.calibrator.fit(X, y)
        else:
            X = check_array(X)
            self.classes_ = [0, 1]
            self.clf.fit(X)
            self.calibrator = None

        # Train clf
        self.feature_importances_ = self.compute_feature_importances(X)
        # Return the classifier
        return self

    def feature_importances(self) -> numpy.ndarray:
        """
        Returns feature importances
        :return: a list
        """
        return self.feature_importances_

    def predict(self, X) -> numpy.ndarray:
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        return numpy.asarray(self.classes_[numpy.argmax(probas, axis=1)])

    def predict_proba(self, X) -> numpy.ndarray:
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        if self.clf is not None:
            if self.calibrator is not None:
                check_is_fitted(self.calibrator.cal_classifier)
                return self.calibrator.predict_proba(X)
            elif isinstance(self.clf, BaseDetector):
                return self.predict_pyod_proba(X)
            return self.clf.predict_proba(X)
        else:
            return None

    def predict_pyod_proba(self, X) -> numpy.ndarray:
        """
        Method to compute probabilities of predicted classes.
        It has to be overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """
        pred_score = self.clf.decision_function(X)
        probs = numpy.zeros((X.shape[0], 2))
        if isinstance(self.clf.contamination, (float, int)):
            pred_thr = pred_score - self.clf.threshold_
        else:
            pred_thr = 0.5
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
        return probs

    def predict_confidence(self, X) -> numpy.ndarray:
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def compute_feature_importances(self, X) -> numpy.ndarray:
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        elif hasattr(self.clf, 'coef_'):
            return numpy.sum(numpy.absolute(self.clf.coef_), axis=0)
        return numpy.zeros(X.shape[1])

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return get_classifier_name(self.clf)

    def get_params(self, deep=True):
        return {'clf': self.clf}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.clf, parameter, value)
        return self
