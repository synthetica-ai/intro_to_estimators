import tensorflow as tf

params = {'learning_rate': 0.001,
         'feature_names':['fixed_acidity','volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide','density','pH','sulphates','alcohol'],
         'train_batch_size' : 33,
         'valid_batch_size' : 20,
         'test_batch_size' : 17,
         'number_of_hidden_units': [200,180,10], 
         'num_epochs' : 5,
         }






# creating the input function for training, validation, prediction
def my_input_fn(dataframe, labels, batch_size, train):
    
    '''
    dataframe = train_set or valid_ set or test_set
    
    labels = train_y or valid_set or test_set
    
    batch_size = 33 for training or 20 for validation or 17 for test_set   
    '''
    

    # normalization
    df = dataframe.copy()
    min_values = df.min(axis=0)
    max_values = df.max(axis=0)
    max_minus_min = max_values - min_values
    df = df - min_values
    df = df.div(max_minus_min) 
    dataset = tf.data.Dataset.from_tensor_slices((df.values, labels.values))
    if train:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

def my_model_fn(features, labels, mode, params):

    ''' 
    
    The model function of the custom estimator
    
    '''
   
    input_feature = 'x'
    feature_columns =[tf.feature_column.numeric_column(input_feature, shape=(11,))]    
    feature_layer_inputs = { input_feature: tf.keras.Input(shape=(11,), dtype=tf.float32)} 
    
    
    # model's architecture
    
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns,dtype=tf.float64)
    feature_layer_outputs = feature_layer(feature_layer_inputs)
    dense_1 = tf.keras.layers.Dense(220, activation='relu',name='dense_1',dtype=tf.float64)(feature_layer_outputs)
    dense_2 = tf.keras.layers.Dense(350, activation='relu', name='dense_2', dtype=tf.float64)(dense_1)
    dense_3 = tf.keras.layers.Dense(350, activation='relu', name='dense_3', dtype=tf.float64)(dense_2)
    dense_4 = tf.keras.layers.Dense(10, activation='softmax', name='dense_4', dtype=tf.float64)(dense_3)
    model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=dense_4)
    
    
    def my_acc(labels,probs):
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='my_acc')
        acc_metric.update_state(y_true=labels, y_pred=probs)
        return {'acc': acc_metric}
    
    
    def loss():
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probs))
    

    
    with tf.GradientTape() as tape:
        
        tape.watch(model.trainable_variables)
        
        probs = model(features)
        class_index = tf.argmax(probs, axis=1 , output_type=tf.int64)
    
   
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            opt = tf.keras.optimizers.Adam()
            opt.iterations = tf.compat.v1.train.get_global_step()
            gradients = tape.gradient(loss(), model.trainable_variables)
            train_op = opt.apply_gradients(zip(gradients, model.trainable_variables))
            return tf.estimator.EstimatorSpec(mode, loss=loss(), train_op = train_op)
        
        
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = my_acc(labels,probs)
        return tf.estimator.EstimatorSpec(mode,loss=loss(), eval_metric_ops=metrics)



    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'top_class_index' : class_index,
            'probs_of_classes' : probs
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)






def my_serving_input_fn(my_params):
    feature_names = my_params['feature_names']
    receiver_tensors = {}
    input_feature = 'x'
    for names in feature_names:
        max_val = my_params[names][0]
        min_val = my_params[names][1]
        dif = max_val - min_val 
        receiver_tensors[names] = (tf.keras.backend.placeholder(shape=[None,1], dtype=tf.float32, name=names) - min_val)/dif  
    
    
    features = {
        input_feature: tf.concat([
            receiver_tensors['fixed_acidity'],
            receiver_tensors['volatile_acidity'],
            receiver_tensors['citric_acid'],
            receiver_tensors['residual_sugar'],
            receiver_tensors['chlorides'],
            receiver_tensors['free_sulfur_dioxide'],
            receiver_tensors['total_sulfur_dioxide'],
            receiver_tensors['density'],
            receiver_tensors['pH'],
            receiver_tensors['sulphates'],
            receiver_tensors['alcohol'],
        ], axis=1) 
    }
    return  tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=features)