import os
import logging
from tqdm import trange
import tensorflow as tf

from .evaluate_fn import evaluate_sess

def train_sess(sess, model_spec, num_steps, writer, params):
    '''Train the model on the data for one epoch

    Args:
        sess: (tf.Session) indicates current session
        model_spec: (dict) contains all graph operations for training the model
        params: (dict) hyperparameters of the model
        num_steps: (int) Number of batches 
        writer: (tf.summary.FileWriter) writer for storing summaries

    Returns:
        None
    
    '''
    
    #Collect all update ops and metrics
    metrics_update_op = model_spec['metrics_update_op']
    train_op = model_spec['train_op']
    summary_op = model_spec['summary_op']
    loss = model_spec['loss']
    metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()
    
    #Initialize the dataset iterator and metrics local variables
    sess.run(model_spec['iterator_initializer'])
    sess.run(model_spec['metric_initializer_op'])
    
    progress_bar = trange(num_steps, position = 0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for step in progress_bar:
        if step%params['save_frequency']==0:
            _,_,summary,loss_val,global_step_val=sess.run([train_op, metrics_update_op, summary_op, loss, global_step])
            writer.add_summary(summary, global_step_val)
        else:
            _,_,loss_val=sess.run([train_op, metrics_update_op, loss])
        progress_bar.set_postfix(loss=round(loss_val,6))
            
    #Compute metrics over entire training data
    train_metrics_values = sess.run({key : val[0] for key, val in metrics.items()})
    train_metrics_string = ' '.join(['{} : {:.6f}'.format(key, val) for key, val in train_metrics_values.items()])
    logging.info("- Train metrics: "+ train_metrics_string)



def train_and_save(train_model_spec, eval_model_spec, model_dir, params, restore_weights = None):
    '''Train the model and save the weights of last 5 epochs and the best epoch

    Args:
        train_model_spec: (dict) contains all graph operations for training the model
        eval_model_spec: (dict) contains all graph operations for evaluating the model
        model_dir: (string) directory path to store weights and summaries
        params: (dict) hyperparameters of the model
        restore_weights: (string) directory path to restore weights from

    Returns:
        None
    '''
    #Initiliaze the saver
    last_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session() as sess:
        begin_epoch = 0
        sess.run(train_model_spec['variables_init_op'])
        if restore_weights is not None:
            #Restore weights from model_dir/restore_weights
            restore_dir = os.path.join(model_dir, restore_weights)
            logging.info('Restoring weights from {}'.format(restore_dir))
            latest_ckpt = tf.train.latest_checkpoint(restore_dir)
            begin_epoch = int(latest_ckpt.split('-')[-1])
            last_saver.restore(sess, latest_ckpt)
            
        #Create summary writer for training and evaluation
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summary'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summary'), sess.graph)

        if params['best_model_metric'] == 'acc':
            eval_name = 'accuracy'
            eval_comp = '>'
            best_eval_val = 0
        
        else :
            eval_name = 'loss'
            eval_comp = '<'
            best_eval_val = 1e4

        
        for epoch in range(begin_epoch, begin_epoch + params['num_epochs']):
            logging.info('Epoch {}/{}'.format(epoch+1, begin_epoch + params['num_epochs']))
            num_steps = (params['train_size'] + params['batch_size'] - 1)//params['batch_size']
            train_sess(sess, train_model_spec, num_steps, train_writer, params)
            
            num_steps = (params['eval_size'] + params['batch_size'] - 1)//params['batch_size']
            eval_metrics = evaluate_sess(sess, eval_model_spec, num_steps, eval_writer)
            
            last_save_path = os.path.join(model_dir, 'last_weights', 'epoch')
            last_saver.save(sess, last_save_path, global_step = epoch+1)
            
            #Update the weights with current best model
            if eval(str(eval_metrics[eval_name]) +  eval_comp + str(best_eval_val)):
                best_eval_val = eval_metrics[eval_name]
                
                best_save_path = os.path.join(model_dir, 'best_weights', 'epoch')
                best_saver.save(sess, best_save_path, global_step = epoch+1)
                logging.info('- Found new best {}. Saving in {}'.format(eval_name, best_save_path))