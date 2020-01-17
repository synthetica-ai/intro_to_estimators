A file to compare the pros and cons of the three data loading methods

1. tf.data.Dataset.from_tensor_slices(pandas_df)
2. tf.data.Dataset.experimental.make_csv_dataset(csv)
3. tf.data.TextLineDataset(csv)

Optional - lower level experimental.CsvDataset class


## tf.data.Dataset Class Description

[class_definition](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)




### tf.data.Dataset.from_tensor_slices(pandas_df)
[tutorial](https://www.tensorflow.org/tutorials/load_data/pandas)


*Functionality Remarks*


Pros:
- ssd

Cons:
-  


### tf.data.Dataset.experimental.make_csv_dataset(csv)
[tutorial](https://www.tensorflow.org/tutorials/load_data/csv)

[function_definition](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)


The function **Returns**:

A Dataset, where each element is a (features, labels) tuple that corresponds to a batch of batch_size CSV rows. The features dictionary maps feature column names to Tensors containing the corresponding column data, and labels is a Tensor containing the column data for the label column (target column) specified by label_name.

*Functionality Remarks*:

Creates batches(groups of rows) from the csv.

Batching actually groups the Dataset to smaller parts!







Pros:
- column type inference
- batching
- shuffling
- select_columns argument selects subset of columns

Cons:
-    


## tf.data.TextLineDataset


*Questions*

- Reads lines and return them as text?
- Are numbers represented as strings?
- If yes then I need string to float conversion! possible con!


*Functionality Remarks*

* a TextLineDataset yields every line of each file. **con(-)**
    -   Lines such as headers or comments can be removed using the Dataset.skip() or Dataset.filter() transformations.

* numbers as strings?



*Observations*

1.  A different Class than tf.data.Dataset



## tf.data.TextLineDataset Class

[Class Description](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset)
