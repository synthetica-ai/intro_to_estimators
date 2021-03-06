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
-

Cons:
-  


### tf.data.Dataset.experimental.make_csv_dataset(csv)
[tutorial](https://www.tensorflow.org/tutorials/load_data/csv)

[function_definition](https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset)


The function **Returns**:

A Dataset, where each element is a (features, labels) tuple that corresponds to a batch of batch_size CSV rows.

 * features *dictionary* :

    The features dictionary maps feature column names to Tensors containing the corresponding column data.

 * labels *Tensor*:

    labels is a Tensor containing the column data for the label column (target column) specified by label_name.

*Functionality Remarks*:

1. The func creates batches(groups of rows) from the csv.

2. Batching practically groups the Dataset to smaller parts!

3. Shuffling???







Pros:
- column type inference
- batching
- shuffling
- select_columns argument selects subset of columns

Cons:
-    


## tf.data.TextLineDataset Class

[Class Description](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset)

[tutorial_1](https://www.tensorflow.org/tutorials/load_data/text)
[tutorial_2](https://www.tensorflow.org/guide/data#consuming_text_data)


*Functionality Remarks*

* TextLineDataset is designed to create a dataset from one or more text files. Given one or more text filenames, it will produce one string-valued element per line for those files.

    - Practically every line of the file is transformed to a string by the TLD class methods
    - Useful for line-based data (like poetry or erro_logs)
    - How is a line defined in a text file?? (by end of line character??)[possible_answer](https://en.wikipedia.org/wiki/Line_(text_file))
    -

* A TextLineDataset yields every line of each file. **(-)**
    -   Lines such as headers or comments can be removed using the Dataset.skip() or Dataset.filter() transformations.
    -    
* Numbers are presented as strings! **(-)** (conversion to int or float neeeded)



*Observations*

1.  A different Class than tf.data.Dataset




*Questions*

- Reads lines and return them as text?
- Are numbers represented as strings?
- If yes then I need string to float conversion! possible con!







# Problems I Faced


1. Use of Path class for specifying data location path.
Loading data works with data_path as string. Tried to find how it could work with data_path as Path.






# Solutions to Problems

Problem_1

a.  Set all paths as Path objects and use string conversion inside the tf make_csv_dataset method like so:
    ```python

    tf.data.experimental.make_csv_dataset(file_pattern=str(data_path))

    ```

b.  Use glob pattern?? i



# Ideas for Future Work

Problem_1 :
    a.  Regex and parsing from user to suggest a specific csv to create a dataset.
