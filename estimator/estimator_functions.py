import pandas as pd
import tensorflow as tf


def dataset_creation(csv_path, print_flag=False):
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
