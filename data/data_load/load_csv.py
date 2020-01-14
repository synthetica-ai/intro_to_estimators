
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

wine_path = Path("/home/jpriest/Desktop/Synthetica/Project/intro_to_estimators/data/winequality.csv")

# How to load csv data in tensorflow?

# First way: Using pandas dataframe
# 1. Read the csv as a pandas dataframe.
# 2. Convert the dataframe to a dictionary.
# 3. Create tensor slices from the created dictionary.
# Code Implementation


def Dataset_creation(csv_path, print_flag=False):
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


wine_dataset = Dataset_creation(wine_path, True)


# Second way: Using tf.data.experimental.make_csv_dataset function
# The second way is useful for scaling up to a large set of files or when one needs a loader that integrates with Tensorflow and the tf.data API
# The implementation involves using the tf.data.experimental.make_csv_dataset function.
# The only column you need to identify explicitly is the one with the value that the model is intended to predict (named label_column here).
# wines_df['quality'].value_counts()
# LABEL_COLUMN = 'quality'
# LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # possible labels from 3 to 8


def get_dataset(file_path, label_column, **kwargs):
    """
    Creates a tf dataset from csv in file_path.

    file_path : The path to the csv file including the csv.

    label_columns : The target column label.

    """

    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        **kwargs)

    return dataset


wine_tf_data = get_dataset(file_path=wine_path, label_column='quality')


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


# TODO  Third Way : Using tf.data.TextLineDatasets
# TODO:  Split the dataset
# Two options:
# 1. Load whole dataset and split with dataset.take(), dataset.skip()
# 2. Split whole dataset before hand maybe with scikit learn?
