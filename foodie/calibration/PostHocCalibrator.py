import numpy
from netcal.binning import HistogramBinning, BBQ
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted, check_array

from foodie.calibration.CalibrationMetric import CalibrationMetric


class PostHocCalibrator:
    """
    Class for building objects for post-hoc calibration strategies.
    It applies only to supervised learning, or at least requires a labelled validation set
    """

    def __init__(self, clf):
        """
        Constructor
        :param clf: the uncalibrated classifier
        """
        self.clf = clf
        self.cal_classifier = None

    def fit(self, x_train, y_train: numpy.ndarray):
        """
        Makes the prediction calibration strategy ready to be applied.
        :param x_train: the train features
        :param y_train: the train labels
        :return:
        """
        pass

    def predict(self, x_test) -> numpy.ndarray:
        """
        Applies the prediction rejection strategy to a specific test set
        :param x_test: the data to apply the strategy to
        :return:
        """
        if self.cal_classifier is None:
            return None
        check_is_fitted(self.cal_classifier)
        x_test = check_array(x_test)
        return self.cal_classifier.predict(x_test)

    def predict_proba(self, x_test) -> numpy.ndarray:
        """
        Method to compute probabilities of predicted classes
        :param x_test: the data to apply the strategy to
        :return: array of probabilities for each classes
        """
        if self.cal_classifier is None:
            return None
        check_is_fitted(self.cal_classifier)
        x_test = check_array(x_test)
        return self.cal_classifier.predict_proba(x_test)

    def get_name(self) -> str:
        """
        Returns the name of the strategy
        :return:
        """
        return self.__class__.__name__

    def describe(self) -> str:
        """
        Returns a textual description of the rejection strategy
        :return: a string
        """
        return ""

    def compute_metric(self, metric: CalibrationMetric, x_test, y_test):
        """
        Scores a specific calibration metric on this calibrator
        :param metric: the metric to be computed
        :param x_test: the dataset to use for scoring
        :param y_test: the labels to use for scoring
        :return:
        """
        check_is_fitted(self.clf)
        check_is_fitted(self.cal_classifier)
        x_test = check_array(x_test)
        return metric.quantify(self.clf, self.cal_classifier, x_test, y_test)


class PlattScaling(PostHocCalibrator):
    """
    Computes calibration using Platt scaling
    """

    def __init__(self, clf):
        """
        Constructor
        :param clf: the uncalibrated classifier
        """
        PostHocCalibrator.__init__(self, clf)

    def fit(self, x_train, y_train: numpy.ndarray):
        """
        Makes the prediction calibration strategy ready to be applied.
        :param x_train: the train features
        :param y_train: the train labels
        :return:
        """
        self.cal_classifier = CalibratedClassifierCV(self.clf, cv='prefit', method='sigmoid')
        self.cal_classifier.fit(x_train, y_train)


class IsotonicScaling(PostHocCalibrator):
    """
    Computes calibration using Isotonic scaling
    """

    def __init__(self, clf):
        """
        Constructor
        :param clf: the uncalibrated classifier
        """
        PostHocCalibrator.__init__(self, clf=clf)

    def fit(self, x_train: numpy.ndarray, y_train: numpy.ndarray):
        """
        Makes the prediction calibration strategy ready to be applied.
        :param clf: the uncalibrated classifier
        :param x_train: the train features
        :param y_train: the train labels
        :return:
        """
        self.cal_classifier = CalibratedClassifierCV(self.clf, cv='prefit', method='isotonic')
        self.cal_classifier.fit(x_train, y_train)


class HistogramScaling(PostHocCalibrator):
    """
    Computes Calibration using Histogram Binning (from netcal package)
    Works only for binary classification
    """

    def __init__(self, clf, n_bins: int = 10, threshold: float = 0.5, labels: list = [0, 1]):
        """
        Constructor
        :param clf: the uncalibrated classifier
        """
        PostHocCalibrator.__init__(self, clf=clf)
        self.n_bins = n_bins
        self.threshold = threshold
        self.labels = labels

    def fit(self, x_train: numpy.ndarray, y_train: numpy.ndarray):
        """
        Makes the prediction calibration strategy ready to be applied.
        :param clf: the uncalibrated classifier
        :param x_train: the train features
        :param y_train: the train labels
        :return:
        """
        proba = self.clf.predict_proba(x_train)
        # In this case the cal_classifier is the HistBin
        self.cal_classifier = HistogramBinning(bins=self.n_bins)
        self.cal_classifier.fit(proba[:, 1], y_train)

    def predict(self, x_test: numpy.ndarray):
        """
        Applies the prediction rejection strategy to a specific test set
        :param x_test: the data to apply the strategy to
        :return:
        """
        if self.cal_classifier is None:
            return None
        probas = self.predict_proba(x_test)
        y_pred = 1*(probas[:, 1] > self.threshold)
        return numpy.asarray(self.labels)[y_pred]

    def predict_proba(self, x_test: numpy.ndarray):
        """
        Method to compute probabilities of being class0 (1D array)
        :param x_test: the data to apply the strategy to
        :return: array of probabilities for each classes
        """
        if self.cal_classifier is None:
            return None
        probas = self.clf.predict_proba(x_test)
        hist_p = self.cal_classifier.transform(probas[:, 1])
        hist_probas = numpy.vstack([1-hist_p, hist_p]).T
        return hist_probas


class BBQScaling(PostHocCalibrator):
    """
    Computes Calibration using BBQ Scaling (from netcal package)
    Works only for binary classification
    """

    def __init__(self, clf, threshold: float = 0.5, labels: list = [0, 1]):
        """
        Constructor
        :param clf: the uncalibrated classifier
        """
        PostHocCalibrator.__init__(self)
        self.threshold = threshold
        self.labels = labels

    def fit(self, clf, x_train: numpy.ndarray, y_train: numpy.ndarray):
        """
        Makes the prediction calibration strategy ready to be applied.
        :param clf: the uncalibrated classifier
        :param x_train: the train features
        :param y_train: the train labels
        :return:
        """
        proba = clf.predict_proba(x_train)
        # In this case the cal_classifier is the BBQ
        self.cal_classifier = BBQ()
        self.cal_classifier.fit(proba[:, 1], y_train)

    def predict(self, x_test: numpy.ndarray):
        """
        Applies the prediction rejection strategy to a specific test set
        :param x_test: the data to apply the strategy to
        :return:
        """
        if self.cal_classifier is None:
            return None
        probas = self.predict_proba(x_test)
        y_pred = 1*(probas[:, 1] > self.threshold)
        return numpy.asarray(self.labels)[y_pred]

    def predict_proba(self, x_test: numpy.ndarray):
        """
        Method to compute probabilities of being class0 (1D array)
        :param x_test: the data to apply the strategy to
        :return: array of probabilities for each classes
        """
        if self.cal_classifier is None:
            return None
        probas = self.clf.predict_proba(x_test)
        bbq_p = numpy.clip(self.cal_classifier.transform(probas[:, 1]), 0, 1)
        bbq_probas = numpy.vstack([1 - bbq_p, bbq_p]).T
        return bbq_probas
