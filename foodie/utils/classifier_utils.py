import numpy
import sklearn

from foodie.calibration.CalibrationMetric import ExpectedCalibrationError, MaximumCalibrationError
from foodie.calibration.PostHocCalibrator import PostHocCalibrator


def evaluate_predictions(confidence: numpy.ndarray, labels: numpy.ndarray, y_test: numpy.ndarray) -> dict:
    """
    Function to evaluate predictions of a classifier on a test set
    :param confidence: the predicted probabilities
    :param labels: predicted labels
    :param y_test: the test labels
    :return: dictionary containing all available metrics for the classifier
    """
    # Multi-class metrics: these apply to any classification problem
    met_dict = {'acc': sklearn.metrics.accuracy_score(y_test, labels),
                'mcc': sklearn.metrics.matthews_corrcoef(y_test, labels),
                'b_acc': sklearn.metrics.balanced_accuracy_score(y_test, labels)}

    # Metrics that use confidence
    met_dict['avg_confidence'] = numpy.average(confidence)
    hit_conf = confidence[y_test == labels]
    met_dict['avg_hit_confidence'] = numpy.average(hit_conf)
    met_dict['min_hit_confidence'] = numpy.min(hit_conf)
    met_dict['max_hit_confidence'] = numpy.max(hit_conf)
    misc_conf = confidence[y_test != labels]
    met_dict['avg_misc_confidence'] = numpy.average(misc_conf)
    met_dict['min_misc_confidence'] = numpy.min(misc_conf)
    met_dict['max_misc_confidence'] = numpy.max(misc_conf)
    met_dict['avg_hit_misc_diff'] = met_dict['avg_hit_confidence'] - met_dict['avg_misc_confidence']
    met_dict['max_min_diff'] = met_dict['max_misc_confidence'] - met_dict['min_hit_confidence']

    if len(numpy.unique(y_test) <= 2) and len(numpy.unique(labels) <= 2):
        # This means it is a binary classification problem
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, labels).ravel()
        met_dict['tn'] = tn
        met_dict['tp'] = tp
        met_dict['fn'] = fn
        met_dict['fp'] = fp
        met_dict['rec'] = sklearn.metrics.recall_score(y_test, labels, zero_division=0)
        met_dict['prec'] = sklearn.metrics.precision_score(y_test, labels, zero_division=0)

    return met_dict


def evaluate_classifier(classifier, x_test, y_test: numpy.ndarray):
    """
    Function to evaluate a classifier on a test set
    :param classifier: the classifier object
    :param x_test: the test set
    :param y_test: the test labels
    :return: dictionary containing all available metrics for the classifier
    """
    confidence = get_confidence(classifier, x_test)
    labels = classifier.predict(x_test)
    met_dict = evaluate_predictions(confidence, labels, y_test)

    # Metrics to evaluate the impact of calibration
    if hasattr(classifier, "calibrator") and isinstance(classifier.calibrator, PostHocCalibrator):
        met_dict["calibration"] = classifier.calibrator.get_name()
        met_dict["ECE"] = classifier.calibrator.compute_metric(ExpectedCalibrationError(), x_test, y_test)
        met_dict["MCE"] = classifier.calibrator.compute_metric(MaximumCalibrationError(), x_test, y_test)
    else:
        met_dict["calibration"] = "none"
        met_dict["ECE"] = None
        met_dict["MCE"] = None

    # Metrics for diversity of ensembles
    if hasattr(classifier, "estimators_"):
        base_learners = classifier.estimators_

    return met_dict


def get_confidence(classifier, x_test) -> numpy.ndarray:
    """
    Method to compute confidence in the predicted class
    :return: max probability as default
    """
    if callable(getattr(classifier, "predict_confidence", None)):
        return classifier.predict_confidence(x_test)
    else:
        probas = classifier.predict_proba(x_test)
        return numpy.max(probas, axis=1)

