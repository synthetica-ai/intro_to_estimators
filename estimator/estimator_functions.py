import pandas as pd
import tensorflow as tf


def dataset_pandas_creation(csv_path, print_flag=False):
    """
    Creating a tf Dataset from csv in csv_path.
    The csv has the target label in the last column.
    If print_flag = True, the Dataset is printed as well.

    Args:
        csv_path (Path): Path to the csv file.

    Returns:
        Dataset: a tf Dataset

    """
    df = pd.read_csv(csv_path)
    target_label = df.columns[-1]
    target_srs = df.pop(target_label)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target_srs.values))
    if print_flag:
        for feature, target in dataset:
            print('Features: {}, Target: {}'.format(feature, target))
    return dataset


def dataset_experimental_csv(csv_path, **kwargs):
    """
    Creates a tf dataset from csv in file_path.

    file_path : The path to the csv file.

    label_columns : The target column label.

    """

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_path,
        batch_size=5,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)

    return dataset


def dataset_textline_creation(data_path):
    """
    A function that creates a tf.TextLineDataset from a csv file in data_path


    Args:
        data_path (type): the path to the dataset

    Returns:
        type: A tf.TextLineDataset

    Raises:
        Exception: description

    """
    str_data_path = str(data_path)
    dataset_lines = tf.data.TextLineDataset(str_data_path)
    return dataset_lines
