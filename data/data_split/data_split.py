# TODO:  Split the dataset
# Two options:
# 1. Load whole dataset first and then split it with dataset.take(), dataset.skip() (+ only need tf module)
# 2. Split whole dataset before loading (creating 3 csvs with a command line tool?) and creating 3 separate tf.Datasets for train, eval, test?(-preprocessing , - loading 3 csvs,  +???? .maybe only option if dataset is huge and cannot be loaded all at once! )

# Pros and Cons of each method
# How to insert randomness in data_split???
