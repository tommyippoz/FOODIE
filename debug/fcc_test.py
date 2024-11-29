import csv
import os

from foodie.classifiers.Classifier import get_classifier_name
from foodie.dataloader.TabularDataLoader import SingleTabularDataLoader
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
LABEL_ENCODING = False
SHUFFLE = False
NORMAL_TAG = None
REMOVE_CATEGORICAL = True
FEATURES_TO_REMOVE = None


# Builds classifiers to be tested
def get_test_classifiers() -> list:
    return []


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

            print("Processing Dataset '%s'", dataset_path)
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
            for clf in get_test_classifiers():
                clf_name = get_classifier_name(clf)
                print("\nBuilding classifier: %s", clf_name)
                start_ms = current_ms()
                clf.fit(x_train, y_train)
                train_ms = current_ms()
                y_pred = clf.predict(x_test)
                test_ms = current_ms() - train_ms
                metrics = evaluate_classifier(classifier=clf, x_test=x_test, y_test=y_test, print_summary=True)
                metrics["train_ms"] = train_ms
                metrics["test_ms"] = test_ms
                metrics["clf_name"] = clf_name
                metrics["dataset_name"] = filename

                # Write to CSV
                with open(OUT_FILE, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
                    if create_file:
                        writer.writeheader()
                        create_file = False
                    writer.writerow(metrics)
