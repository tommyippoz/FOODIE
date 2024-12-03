from netcal.metrics import ECE, MCE


class CalibrationMetric:
    """
    Used to quantify the advantages of using a calibration strategy
    The most common metric to determine miscalibration in the scope of classification is the Expected Calibration Error (ECE).
    This metric divides the confidence space into several bins and measures the observed accuracy in each bin.
    The bin gaps between observed accuracy and bin confidence are summed up and weighted by the amount of samples in each bin.
    The Maximum Calibration Error (MCE) denotes the highest gap over all bins.
    The Average Calibration Error (ACE) denotes the average miscalibration where each bin gets weighted equally.
    """

    def __init__(self):
        """
        Default Constructor
        """
        pass

    def quantify_predictions(self, uncal_proba, cal_proba, label) -> float:
        """
        Function to be overridden in superclasses
        :param label: the ground truth label
        :param uncal_proba: probabilities predicted for uncalibrated classifier
        :param cal_proba: probabilities predicted for the calibrated classifier
        :return: the metric value (float)
        """
        pass

    def quantify(self, clf, cal_clf, test_data, label) -> float:
        """
        Function used to compute the metric starting from classifiers and data
        :param label: the ground truth label
        :param clf: the uncalibrated classifier object
        :param cal_clf: the calibrated classifier object
        :param test_data: the test data to be used for metric
        :return: the metric value (float)
        """
        return self.quantify_predictions(clf.predict_proba(test_data), cal_clf.predict_proba(test_data), label)


class ExpectedCalibrationError(CalibrationMetric):
    """
    Computes the ECE from netcal package
    """

    def __init__(self, n_bins:int=10):
        CalibrationMetric.__init__(self)
        self.n_bins = n_bins

    def quantify_predictions(self, uncal_proba, cal_proba, label) -> float:
        """
        Function to be overridden in superclasses
        :param label: the ground truth label
        :param uncal_proba: probabilities predicted for uncalibrated classifier
        :param cal_proba: probabilities predicted for the calibrated classifier
        :return: the metric value (float)
        """
        ece = ECE(self.n_bins)
        uncalibrated_score = ece.measure(uncal_proba, label)
        calibrated_score = ece.measure(cal_proba, label)
        return (uncalibrated_score - calibrated_score) / uncalibrated_score if uncalibrated_score > 0 else 0.0


class MaximumCalibrationError(CalibrationMetric):
    """
    Computes the MCE from netcal package
    """

    def __init__(self, n_bins=10):
        CalibrationMetric.__init__(self)
        self.n_bins = n_bins

    def quantify_predictions(self, uncal_proba, cal_proba, label) -> float:
        """
        Function to be overridden in superclasses
        :param label: the ground truth label
        :param uncal_proba: probabilities predicted for uncalibrated classifier
        :param cal_proba: probabilities predicted for the calibrated classifier
        :return: the metric value (float)
        """
        mce = MCE(self.n_bins)
        uncalibrated_score = mce.measure(uncal_proba, label)
        calibrated_score = mce.measure(cal_proba, label)
        return (uncalibrated_score - calibrated_score) / uncalibrated_score if uncalibrated_score > 0 else 0.0
