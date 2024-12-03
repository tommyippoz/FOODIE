from collections.abc import Iterable

import numpy
import pandas
import sklearn

from foodie.dataloader.DataLoader import DataLoader


class TabularDataLoader(DataLoader):
    """
    DataLoader for Tabular Data
    """

    def __init__(self, label_name: str, data_limit: int = None, tt_split: float = 0.5, tv_split: float = None,
                 label_encoding: bool = False, shuffle: bool = False, normal_tag: str = None):
        """
        Constructor of a TabularDataLoader
        """
        DataLoader.__init__(self, data_limit, tt_split, tv_split, label_encoding, shuffle, normal_tag)
        self.label_name = label_name
        self.feature_names = None
        self.label_names = None

    def get_feature_names(self) -> list:
        """
        Returns the tags of features in the dataset
        :return:
        """
        return self.feature_names

    def get_unique_labels(self) -> Iterable:
        """
        Returns the unique labels in the dataset
        :return:
        """
        return self.label_names


class SingleTabularDataLoader(TabularDataLoader):
    """
    DataLoader for Tabular data when all data is contained in a single file
    """

    def __init__(self, tab_filename: str, label_name: str, data_limit: int = None,
                 remove_categorical: bool = True, remove_columns=None,
                 tt_split: float = 0.5, tv_split: float = None,
                 label_encoding: bool = False, shuffle: bool = False, normal_tag: str = None):
        """
        Constructor of a SingleTabularDataLoader
        """
        TabularDataLoader.__init__(self, label_name, data_limit, tt_split, tv_split, label_encoding, shuffle, normal_tag)
        self.remove_columns = remove_columns
        self.remove_categorical = remove_categorical
        self.tab_filename = tab_filename

    def load_data(self) -> None:
        """
        Loads data from specified source
        """

        # Loading Dataset
        df = pandas.read_csv(self.tab_filename, sep=",")

        # Shuffle
        if self.shuffle:
            df = df.sample(frac=1.0)
        df = df.fillna(0)
        df = df.replace('null', 0)

        # Cut data if it exceeds the data_limit
        if (self.data_limit is not None) and (numpy.isfinite(self.data_limit)) and (self.data_limit < len(df.index)):
            df = df[0:self.data_limit]

        # Checks if label column exists in file
        if self.label_name in df.columns:

            # Checks if labels need to be "binarized"
            if self.normal_tag is not None:
                df[self.label_name] = numpy.where(df[self.label_name] == self.normal_tag, "normal", "anomaly")

            # Checks if label has to be encoded
            if self.label_encoding:
                encoding, mapping = pandas.factorize(df[self.label_name])
                self.normal_tag = list(mapping).index(self.normal_tag)
                y = numpy.asarray(encoding)
                self.label_names = list(mapping)

            else:
                y = df[self.label_name].to_numpy()
                self.label_names = numpy.unique(y)

            print("Dataset loaded: " + str(len(df.index)) + " items and " + str(len(self.label_names)) + " labels")
            x = df.drop(columns=[self.label_name])
        else:
            print("Unable to find label column '%s'" % self.label_name)
            x = df
            y = None

        # If columns have to be removed by default
        if self.remove_columns is not None and isinstance(self.remove_columns, Iterable):
            t_remove = [str(col) for col in self.remove_columns if str(col) in x.columns]
            x = x.drop(columns=t_remove)

        # If categorical values have to be dropped
        if self.remove_categorical:
            x = x.select_dtypes(exclude=['object'])

        # Final operations and Splitting
        self.feature_names = x.columns
        if self.tt_split is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                sklearn.model_selection.train_test_split(x, y, test_size=1 - self.tt_split, shuffle=self.shuffle)
            if self.tv_split is not None:
                self.x_train, self.x_val, self.y_train, self.y_val = \
                    sklearn.model_selection.train_test_split(self.x_train, self.y_train,
                                                             test_size=1 - self.tv_split,
                                                             shuffle=self.shuffle)
        else:
            self.x_train = x
            self.y_train = y
