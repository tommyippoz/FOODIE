from foodie.classifiers.Classifier import Classifier


class FailControlledClassifier(Classifier):
    """
    Base object for FCCs
    """

    def __init__(self, clf, rejection_adjudicator, alr: float, rejectors: list = None, calibrator_str: str = None, reject_tag:str = None):
        """
        Constructor of a generic Classifier
        :param rejection_adjudicator: the function that decides on rejections when multiple rejectors are used
        :param reject_tag: the tag to be used as output when rejecting predictions
        :param rejectors: the list of prediction rejectors
        :param alr: the acceptable level of risk (i.e., misclassifications) for this FCC
        :param clf: algorithm to be used as Classifier
        :param calibrator_str: post-hoc calibration strategy to be used in the Classifier
        """
        Classifier.__init__(clf, calibrator_str)
        self.rejection_adjudicator = rejection_adjudicator
        self.reject_tag = reject_tag
        self.rejectors = rejectors
        self.alr = alr



