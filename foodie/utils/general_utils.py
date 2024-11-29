import time


def is_csv_file(filename) -> bool:
    """
    True if the file is a CSV file
    :param filename: string path to file
    :return: boolean
    """
    return filename.endswith(".csv")


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)
