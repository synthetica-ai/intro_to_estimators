A file to compare the pros and cons of the three loading methods

1. pandas
2. experimental.make_csv_dataset
3. TextLineDatasets

Optional - lower level experimental.CsvDataset class




## pandas
[link](https://www.tensorflow.org/tutorials/load_data/pandas)

Pros:
    -

Cons:
    -  


## experimental.make_csv_dataset
[link](https://www.tensorflow.org/tutorials/load_data/csv)
[function_definition](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)


The class **Returns**:
A dataset, where each element is a (features, labels) tuple that corresponds to a batch of batch_size CSV rows. The features dictionary maps feature column names to Tensors containing the corresponding column data, and labels is a Tensor containing the column data for the label column (target column) specified by label_name.

Remarks:

Creates batches(groups of rows) from the csv.

Batching actually groups the dataset to smaller parts!







Pros:
    - column type inference
    - batching
    - shuffling
    - select_columns argument selects subset of columns

Cons:
    -    






## Dataset Class Description

[class_definition](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
