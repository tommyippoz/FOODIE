import csv
import os
import warnings

from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

from foodie.classifiers.Classifier import get_classifier_name, Classifier
from foodie.dataloader.TabularDataLoader import SingleTabularDataLoader
from foodie.utils.classifier_utils import evaluate_classifier
from foodie.utils.general_utils import is_csv_file, current_ms

# Paths
DATA_FOLDER = 'input_folder/tabular'
OUTPUT_FOLDER = 'output_folder'
OUT_FILE = 'test_results.csv'

# Additional Data
LABEL_NAME = 'multilabel'
ROW_LIMIT = None
TT_SPLIT = 0.5
TV_SPLIT = None
LABEL_ENCODING = True
SHUFFLE = True
NORMAL_TAG = "normal"
REMOVE_CATEGORICAL = True
FEATURES_TO_REMOVE = ['timestamp', 'time', 'ip', 'src_ip', 'dst_ip', 'port', 'src_port', 'dst_port']


def get_test_classifiers(contamination: float) -> list:
    """
    Gets classifiers to be tested as a list
    :param contamination: a float value indicating the fractio of anomalies in the train/val set, needed for PYOD
    :return: a list of classifiers
    """
    classifiers = [
        LinearDiscriminantAnalysis(),
        Classifier(clf=LinearDiscriminantAnalysis()),
        Classifier(clf=LinearDiscriminantAnalysis(), calibrator_str='platt'),
        Classifier(clf=LinearDiscriminantAnalysis(), calibrator_str='isotonic'),
        XGBClassifier(n_estimators=10),
        Classifier(clf=XGBClassifier(n_estimators=10)),
        Classifier(clf=XGBClassifier(n_estimators=10), calibrator_str='platt'),
        Classifier(clf=XGBClassifier(n_estimators=10), calibrator_str='isotonic')
    ]
    if contamination > 0:
        if contamination > 0.5:
            contamination = 0.5
            print("Warning: amount of anomalies in the dataset is bigger than 50%")
        classifiers.extend([Classifier(COPOD(contamination=contamination)),
                            COPOD(contamination=contamination),
                            Classifier(PCA(contamination=contamination)),
                            PCA(contamination=contamination)
                            ])
    return classifiers


if __name__ == '__main__':
    """
    Main to debug-test FOODIE
    """

    # Check for input folder
    if not os.path.exists(DATA_FOLDER):
        print("Unable to find data folder '%s'", DATA_FOLDER)
        exit(1)

    # Removes existing output file
    if os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
    create_file = True

    # Iterating over CSV files in folder
    for filename in os.listdir(DATA_FOLDER):
        dataset_path = os.path.join(DATA_FOLDER, filename)
        if os.path.isfile(dataset_path) and is_csv_file(dataset_path):

            print("Processing Dataset '%s'" % dataset_path)
            data_loader = SingleTabularDataLoader(tab_filename=dataset_path, label_name=LABEL_NAME,
                                                  data_limit=ROW_LIMIT, remove_categorical=REMOVE_CATEGORICAL,
                                                  remove_columns=FEATURES_TO_REMOVE,
                                                  tt_split=TT_SPLIT, tv_split=TV_SPLIT,
                                                  label_encoding=LABEL_ENCODING, shuffle=SHUFFLE, normal_tag=NORMAL_TAG)
            data_loader.load_data()

            # Get Dataset partitions for experiments
            x_train, y_train = data_loader.get_train_data()
            x_val, y_val = data_loader.get_validation_data()
            x_test, y_test = data_loader.get_test_data()

            # Iterating over classifiers
            for clf in get_test_classifiers(data_loader.get_contamination()):
                clf_name = get_classifier_name(clf)
                start_ms = current_ms()
                with warnings.catch_warnings():
                    # This is to avoid prompting the PYOD warning "y should not be presented in unsupervised learn.."
                    warnings.simplefilter("ignore")
                    clf.fit(x_train, y_train)
                train_ms = current_ms()
                y_pred = clf.predict(x_test)
                test_time = current_ms() - train_ms
                metrics = evaluate_classifier(classifier=clf, x_test=x_test, y_test=y_test)
                metrics["train_ms"] = train_ms - start_ms
                metrics["test_ms"] = test_time
                metrics["clf_name"] = clf_name
                metrics["dataset_name"] = filename
                print("\tClassifier '%s' has accuracy of %.3f, train/test in %d/%d milliseconds" %
                      (clf_name, metrics["acc"], metrics["train_ms"], metrics["test_ms"]))

                # Write to CSV
                with open(OUT_FILE, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
                    if create_file:
                        writer.writeheader()
                        create_file = False
                    writer.writerow(metrics)
