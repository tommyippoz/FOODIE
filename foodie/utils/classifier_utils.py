from foodie.classifiers import Classifier


def get_classifier_name(clf) -> str:
    """
    Reads the object and extracts a name for such a classifier
    :param clf: the classifier object
    :return:
    """
    if isinstance(clf, Classifier):
        return clf.classifier_name()
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
