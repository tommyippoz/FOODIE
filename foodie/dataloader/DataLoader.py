from collections.abc import Iterable


class DataLoader:

    def __init__(self, data_limit: int = None, tt_split: float = 0.5, tv_split: float = None,
                 label_encoding: bool = False, shuffle: bool = False, normal_tag: str = None):
        """
        Constructor of a generic DataLoader
        :param normal_tag: the name of the normal class, if any. If None, leaves labels as they are.
            Otherwise makes the problem binary (normal vs anomaly)
        """
        self.normal_tag = normal_tag
        self.shuffle = shuffle
        self.label_encoding = label_encoding
        self.tv_split = tv_split
        self.tt_split = tt_split
        self.data_limit = data_limit
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.feature_list = None

    def get_test_data(self):
        """
        Gets test data, both features and labels (if any)
        :return:
        """
        return self.x_test, self.y_test

    def get_train_data(self):
        """
        Gets train data, both features and labels (if any)
        :return:
        """
        return self.x_train, self.y_train

    def get_validation_data(self):
        """
        Gets train data, both features and labels (if any)
        :return:
        """
        return self.x_val, self.y_val

    def get_feature_number(self) -> int:
        """
        Returns the number of features in the dataset
        :return:
        """
        return len(self.get_feature_names())

    def get_n_classes(self) -> int:
        """
        Returns the number of classes of the problem
        :return:
        """
        return len(self.get_unique_labels())

    def get_contamination(self) -> float:
        """
        Returns the percentage of anomalies in the train set (if labeled) or validation (if any)
        :return: the contamination value
        """
        if self.normal_tag is not None:
            if self.y_train is not None:
                return 1 - (self.y_train == self.normal_tag).sum()/len(self.y_train)
            elif self.y_val is not None:
                return 1 - (self.y_val == self.normal_tag).sum() / len(self.y_val)
            else:
                return 0.0
        else:
            return 0.0

    # ------------------------ TO BE OVERRIDDEN --------------------------------

    def load_data(self) -> None:
        """
        Function to be called for fetching data and initialize train, validation, test sets
        Loads data from specified source, to be overridden by superclasses
        :return:
        """
        pass

    def get_feature_names(self) -> list:
        """
        Returns the tags of features in the dataset
        :return:
        """
        pass

    def get_unique_labels(self) -> Iterable:
        """
        Returns the unique labels in the dataset
        :return:
        """
        pass

