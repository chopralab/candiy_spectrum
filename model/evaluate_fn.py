import os
import logging
from tqdm import trange

import tensorflow as tf

def evaluate_sess(sess, model_spec, num_steps, writer, params):
    '''Evaluate the model on entire validation data

    Args:
        sess: (tf.Session) indicates current session
        model_spec: (dict) contains all graph operations for evaluating the model
        params: (dict) hyperparameters of the model
        num_steps: (int) Number of batches 
        writer: (tf.summary.FileWriter) writer for storing summaries, can be None

    Returns:
        eval_metrics_values: (string) contains metrics of validation data
    
    '''
    
    #Collect all operations for evaluation
    metrics_update_op = model_spec['metrics_update_op']
    metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()
    
    #Initiliaze the dataset iterator and metrics local variables
    sess.run(model_spec['initializer'])
    sess.run(model_spec['metric_initializer_op'])
    
    progress_bar = trange(num_steps, position = 0)
    for _ in progress_bar:
        _ = sess.run(metrics_update_op)
    
    #Compute and log evaluation metrics
    eval_metrics_values = sess.run({key : val[0] for key, val in metrics.items()})
    eval_metrics_string = ''.join(['{} : {:.3f}'.format(key, val) for key, val in eval_metrics_values.items()])
    logging.info("- Eval metrics: "+ eval_metrics_string)
    
    #Add evaluation summaries to the writer
    if writer is not None:
        global_step_val = sess.run(global_step)
        for key, val in eval_metrics_values.items():
            summary = tf.Summary(value = [tf.Summary.Value(tag = key, simple_value = val)])
            writer.add_summary(summary, global_step = global_step_val)
    return eval_metrics_values

def evaluate(model_spec, model_dir, params, restore_weights = None):
    '''Evaluate the model after restoring the weights

    Args:
        model_spec: (dict) contains all graph operations for training the model
        model_dir: (string) directory path to restore weights and write summaries
        params: (dict) hyperparameters of the model
        restore_weights: (string) directory path to restore weights from

    Returns:
        None
    '''

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        #Restore weights from model_dir/restore_weights
        restore_dir = os.path.join(model_dir, restore_weights)
        logging.info('Restoring weights from {}'.format(restore_dir))
        latest_ckpt = tf.train.latest_checkpoint(restore_dir)
        saver.restore(sess, latest_ckpt)

        num_steps = (params['eval_size'] + params['batch_size'] - 1)//params['batch_size']
        _ = evaluate_sess(sess, model_spec, num_steps, None, params)
            
