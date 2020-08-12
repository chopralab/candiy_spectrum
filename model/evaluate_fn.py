import os
import logging
from tqdm import trange

import numpy as np
import tensorflow as tf

def evaluate_sess(sess, model_spec, num_steps, writer, feed_dict = {}):
    '''Evaluate the model on entire data

    Args:
        sess: (tf.Session) indicates current session
        model_spec: (dict) contains all graph operations for evaluating the model
        num_steps: (int) Number of batches 
        writer: (tf.summary.FileWriter) writer for storing summaries, can be None
        feed_dict: (dict) containing mode during evaluation

    Returns:
        eval_metrics_values: (string) contains evaluation metrics of data
    
    '''
    
    #Collect all operations for evaluation
    metrics_update_op = model_spec['metrics_update_op']
    metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()
    
    #Initiliaze the dataset iterator and metrics local variables
    sess.run(model_spec['iterator_initializer'])
    sess.run(model_spec['metric_initializer_op'])
    
    progress_bar = trange(num_steps, position = 0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for _ in progress_bar:
        _ = sess.run(metrics_update_op, feed_dict = feed_dict)
    
    #Compute and log evaluation metrics
    eval_metrics_values = sess.run({key : val[0] for key, val in metrics.items()})
    eval_metrics_string = ' '.join(['{} : {:.6f}'.format(key, val) for key, val in eval_metrics_values.items()])
    logging.info("- Eval metrics: "+ eval_metrics_string)
    
    #Add evaluation summaries to the writer
    if writer is not None:
        global_step_val = sess.run(global_step)
        for key, val in eval_metrics_values.items():
            summary = tf.Summary(value = [tf.Summary.Value(tag = key, simple_value = val)])
            writer.add_summary(summary, global_step = global_step_val)
    return eval_metrics_values


def predictions_sess(sess, model_spec, size, params, layer_name = 'pred_probs', feed_dict = {}):
    '''Compute predictions of a model layer in model specification

    Args:
        sess: (tf.Session) indicates current session
        model_spec: (dict) contains graph operations for making prediction
        size: (int) dataset size
        params: (dict) hyperparameters of the model
        feed_dict: (dict) containing mode during evaluation


    Returns:
        data: (tuple) containing arrays of prediction and target
    
    '''
    
    #Compute dimension size to create data arrays
    target_dim = model_spec['target'].shape[-1]
    pred_dim = model_spec[layer_name].shape[-1]

    #Initialize target and predictions array
    target_arr = np.zeros((size, target_dim)) 
    pred_arr = np.zeros((size, pred_dim))

    #Initiliaze the dataset iterator
    sess.run(model_spec['iterator_initializer'])
    

    batch_size = params['batch_size']
    #Compute number of batches
    num_steps = (size + batch_size- 1)//batch_size

    #Compute batch wise target and predictions. Add it to data array. 
    progress_bar = trange(num_steps, position = 0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for step in progress_bar:
        pred_batch, target_batch = sess.run([model_spec[layer_name], model_spec['target']], feed_dict = feed_dict)
        target_arr[step*batch_size: (step+1)* batch_size] = target_batch
        pred_arr[step*batch_size: (step+1)* batch_size] = pred_batch

    return pred_arr, target_arr

def evaluate_and_predict(model_spec, layer_name, is_train_data,\
                params, restore_path, evaluate_model = True):
    '''Evaluate the model and make predictions after restoring the weights

    Args:
        model_spec: (dict) contains all graph operations for evaluating the model
        layer_name: (string) name of the layer to compute model predictions
        is_train_data: (bool) whether dataset is train data or val data
        params: (dict) hyperparameters of the model
        restore_path: (string) directory path to restore weights from
        evaluate_model: (bool) whether or not to evaluate the model

    Returns:
        data: (tuple) containing arrays of prediction and target
    '''

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        #Restore weights from model_dir/restore_weights
        restore_dir = os.path.join(restore_path)
        logging.info('Restoring weights from {}'.format(restore_dir))
        latest_ckpt = tf.train.latest_checkpoint(restore_dir)
        saver.restore(sess, latest_ckpt)

        size = params['train_size'] if is_train_data else params['eval_size']
        num_steps = (size + params['batch_size'] - 1)//params['batch_size']

        is_train_ph = model_spec['train_ph']
        feed_dict = {is_train_ph: False}
        
        if evaluate_model:
            _ = evaluate_sess(sess, model_spec, num_steps, None, feed_dict)
        
        
        data = predictions_sess(sess, model_spec, size, params, layer_name, feed_dict)
        return data
            
