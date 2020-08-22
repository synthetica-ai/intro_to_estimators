from estimator.estimator_functions import params, my_input_fn, my_model_fn, my_serving_input_fn
from preprocessing.preprocessing_functions import space_replacer, index_replacer, dataset_split, max_min_finder
import pandas as pd
import tensorflow as tf



wines_df = pd.read_csv("./data/winequality.csv")
space_replacer(wines_df)
index_replacer(wines_df)



dfs, trgts = dataset_split(wines_df)
df_train, df_valid, df_test = dfs
train_target , valid_target, test_target = trgts


params.update(max_min_finder(dataframe=df_train,parameters=params))

data = my_input_fn(dataframe=df_train, labels=train_target, batch_size=params['train_batch_size'], train=True)




configurations = tf.estimator.RunConfig(save_summary_steps=35, save_checkpoints_steps=35, keep_checkpoint_max=1)
my_second_estimator = tf.estimator.Estimator(model_fn = my_model_fn, model_dir='./model/my_second_estimator', params=params, config=configurations)

train_spec = tf.estimator.TrainSpec(input_fn=lambda:my_input_fn(df_train, train_target, params['train_batch_size'], train=True), max_steps=35*params['num_epochs'])

eval_spec = tf.estimator.EvalSpec(input_fn=lambda:my_input_fn(df_valid, valid_target, params['valid_batch_size'], train=False), steps=12)

tf.estimator.train_and_evaluate(my_second_estimator, train_spec, eval_spec)


export_dir = './model/models/wine_model' #rename long number with number of version

serving_input_fn = my_serving_input_fn(my_params=params)

export_path = my_second_estimator.export_saved_model(
    export_dir_base=export_dir, serving_input_receiver_fn=lambda:my_serving_input_fn(params))