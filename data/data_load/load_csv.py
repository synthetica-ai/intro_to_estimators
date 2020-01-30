
# import glob
import os  # os kai pathlib modules for combining changing directories and Path methods and attributes
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Setting data_path as string or Path comparison

data_path_as_string = "/home/jpriest/Desktop/Synthetica/Project/intro_to_estimators/data/winequality.csv"

data_path_as_Path = Path(data_path_as_string)


# How to load csv data in tensorflow?

# First way: Using pandas dataframe
# 1. Read the csv as a pandas dataframe.
# 2. Convert the dataframe to a dictionary.
# 3. Create tensor slices from the created dictionary.
# Code Implementation


def dataset_pandas_creation(data_path, print_flag=False):
    """
    Creating a tf.data.Dataset object from a pandas dataframe.
    The Dataset has the target label in the last column.
    If print_flag = True, the Dataset is printed as well.

    Args:
        data_path (Path or string): Path to the csv file.

    Returns:
        Dataset: a tf Dataset

    """
    df = pd.read_csv(data_path)
    target_label = df.columns[-1]
    target_srs = df.pop(target_label)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target_srs.values))
    if print_flag:
        for feature, target in dataset:
            print('Features: {}, Target: {}'.format(feature, target))
    return dataset


method_one_dataset = dataset_pandas_creation(data_path_as_Path, True)

df = pd.read_csv(data_path_as_Path)

pd.value_counts(df['quality'] == 8)
# Second way: Using tf.data.experimental.make_csv_dataset function
# The second way is useful for scaling up to a large set of files or when one needs a loader that integrates with Tensorflow and the tf.data API
# The implementation involves using the tf.data.experimental.make_csv_dataset function.
# The only column you need to identify explicitly is the one with the value that the model is intended to predict (named label_column here).


label_column = 'quality'


# csv_pattern = "/home/jpriest/Desktop/Synthetica/Project/intro_to_estimators/data/*.csv"


def dataset_experimental_csv(data_pattern, **kwargs):
    """
    Creates a tf dataset from csv in data_path.

    data_path (string) : The path to the csv file as str.

    label_columns : The target column label.

    """

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=str(data_pattern),
        batch_size=5,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)

    return dataset


method_two_dataset = dataset_experimental_csv(data_path_as_Path)

# Important! Loading works with data_path as string and not as Path object. Path object not iterable error! Check problems in md file!


for batch, target_label in method_two_dataset.take(1):
    print(target_label)


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


show_batch(method_two_dataset)


# Third Way : Using tf.data.TextLineDataset

# Needs to load each line of the csv separately and add it to the dataset one by one (-)


# Dataset Creation
def dataset_textline_creating(data_path):
    """
    A function that creates a tf.TextLineDataset from a csv file in data_path


    Args:
        data_path (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    str_data_path = str(data_path)
    dataset_lines = tf.data.TextLineDataset(str_data_path)
    return dataset_lines

# Printing first five lines (each line is a tensor of shape() and dtype=string)
# Eager Execution allows you to conver each line to a numpy array with line.numpy() method


for line in dataset_lines.take(1):
    print(line)


# create func to exclude some lines of the dataset (keep lines with quality=8)


def best_quality(line):
    """
    A short description.

    A bit longer description.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    return tf.equal(tf.strings.substr(line, -1, 1), "8")

# re


best_wines = dataset_lines.skip(1).filter(best_quality)
