'''
Custom Estimator Components

1. input_func : transforms raw data to Dataset objects.

2. feature_func : function that defines the feature cols of the datasets

3. model_func : heart of the estimator. This func specifies the type of model used to make predictions and its characteristics e.g DNN with k layers so on and so forth

4. train_func, eval_func, test_func : functions relevant to implement the training, evaluation and testing procedures.


'''

# input_func


def input_func(dataset):
    ...  # manipulate dataset, extracting the feature dict and the label
    return feature_dict, label


# Define the feature columns including their names and type of data they contain.

def feature_func(arg):

    population = tf.feature_column.numeric_column('population')
    crime_rate = tf.feature_column.numeric_column('crime_rate')
    median_education = tf.feature_column.numeric_column(
        'median_education', normalizer_fn=lambda x: x - global_education_mean)


# Instantiate an estimator, by passing in the feature columns.


def model_func(arg):
    # using premade at first then extend it to custom
    estimator = tf.estimator.LinearClassifier(feature_columns=[population, crime_rate, median_education])


# `input_fn` is the function created in Step 1


def train_func(arg):
    estimator.train(input_func=my_training_set, steps=2000)


def eval_func(arg):
    estimator.eval(input_func=my_eval_set, .....)
    pass


def test_func(arg):
    estimator.test(input_func=my_test_set, .....)
