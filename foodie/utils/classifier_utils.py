import warnings

import numpy
import sklearn
from sklearn.utils.validation import check_is_fitted

from foodie.calibration.CalibrationMetric import ExpectedCalibrationError, MaximumCalibrationError
from foodie.calibration.PostHocCalibrator import PostHocCalibrator
from foodie.classifiers.Classifier import get_classifier_name
from foodie.diversity.EnsembleDiversityMetric import QStatMetric, SigmaMetric, DisagreementMetric, SharedFaultMetric

DIVERSITY_METRICS = [  # QStatMetric(),
    SigmaMetric(),
    DisagreementMetric(relative=True), SharedFaultMetric(relative=True)]


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


def evaluate_classifier(classifier, x_test, y_test: numpy.ndarray, label_tags=None):
    """
    Function to evaluate a classifier on a test set
    :param label_tags: classes of the problem
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
    base_learners = get_baselearners(classifier)
    div_dict, bl_dict = compute_ensemble_diversity(base_learners, x_test, y_test, label_tags)
    met_dict.update(div_dict)

    return met_dict


def get_baselearners(classifier) -> list:
    """
    Gets base-learners (if existing) of a classifier
    :param classifier: the classifier object
    :return: a list of base learners, or an empty list
    """
    if hasattr(classifier, "estimators_"):
        return classifier.estimators_
    elif hasattr(classifier, "clf") and hasattr(classifier.clf, "estimators_"):
        return classifier.clf.estimators_
    else:
        return []


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


def predict_baselearners(base_learners, x_test, label_tags=None, debug: bool = True):
    """
    Predict function to match SKLEARN standards
    :param label_tags: classes of the problem
    :param base_learners: the base_learners to test
    :param test_x: test set
    :return: predictions of individual classifiers, and an array of their names
    """
    bl_predictions = []
    bl_names = []
    for clf in base_learners:
        try:
            check_is_fitted(clf)
            with warnings.catch_warnings():
                # This is to avoid prompting generic warnings while predicting
                warnings.simplefilter("ignore")
                clf_pred = numpy.asarray(clf.predict(x_test))
            if label_tags is not None and len(label_tags) > 0:
                if clf_pred.ndim == 1 and clf_pred[0] not in label_tags:
                    clf_pred = numpy.where(clf_pred < 0.5, label_tags[1], label_tags[0])
                elif clf_pred.ndim == 2 and clf_pred[0, 0] not in label_tags:
                    clf_pred = label_tags[numpy.argmax(clf_pred, axis=1)]
        except:
            return [], []
        bl_predictions.append(clf_pred)
        bl_names.append(get_classifier_name(clf))

    # Cleans up and unifies the dataset of predictions
    bl_predictions = numpy.column_stack(bl_predictions)
    bl_predictions = numpy.nan_to_num(bl_predictions, nan=-0, posinf=0, neginf=0)

    return bl_predictions, bl_names


def compute_ensemble_diversity(base_learners, x_test, y_test, label_tags=None):
    """
    Computes different diversity metrics of bae-learners in the ensemble
    :param base_learners: the list of base-learners
    :param x_test: test set
    :param y_test: test labels
    :return: a dictionary of diversity metrics, and of classification metrics for each base learner
    """
    div_dict = {"n_estimators": len(base_learners)}
    for metric in DIVERSITY_METRICS:
        div_dict[metric.get_name()] = 0.0
    if base_learners is not None and len(base_learners) > 1:
        bl_metrics = {}
        bl_predictions, bl_names = predict_baselearners(base_learners, x_test, label_tags)
        if bl_predictions is not None and len(bl_predictions) > 0:
            for metric in DIVERSITY_METRICS:
                div_dict[metric.get_name()] = metric.compute_diversity(bl_predictions, y_test)
                # print("Diversity using metric " + metric.get_name() + ": " + str(div_dict[metric.get_name()]))
            for i in range(0, bl_predictions.shape[1]):
                bl_metrics["clf_" + str(i)] = {}
                bl_metrics["clf_" + str(i)]["acc"] = sklearn.metrics.accuracy_score(y_test, bl_predictions[:, i])
                bl_metrics["clf_" + str(i)]["b_acc"] = sklearn.metrics.balanced_accuracy_score(y_test,
                                                                                               bl_predictions[:, i])
                bl_metrics["clf_" + str(i)]["mcc"] = sklearn.metrics.matthews_corrcoef(y_test, bl_predictions[:, i])
            bl_metrics["best_base_acc"] = max([bl_metrics[k]["acc"] for k in bl_metrics.keys()])
            bl_metrics["best_base_mcc"] = max(
                [bl_metrics[k]["mcc"] for k in bl_metrics.keys() if isinstance(bl_metrics[k], dict)])
            bl_metrics["best_base_b_acc"] = max(
                [abs(bl_metrics[k]["b_acc"]) for k in bl_metrics.keys() if isinstance(bl_metrics[k], dict)])
    else:
        bl_metrics = {}

    return div_dict, bl_metrics
