import logging
from tqdm import trange

import tensorflow as tf

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
    
    metrics_update_op = model_spec['metrics_update_op']
    train_op = model_spec['train_op']
    summary_op = model_spec['summary_op']
    loss = model_spec['loss']
    global_step = tf.train.get_global_step()
    
    sess.run(model_spec['initializer'])
    sess.run(model_spec['metric_initializer_op'])
    
    progress_bar = trange(num_steps, position = 0)
    for step in progress_bar:
        if step%params['save_frequency']==0:
            _,_,summary,loss_val,global_step_val=sess.run([train_op, metrics_update_op, summary_op, loss, global_step])
            writer.add_summary(summary, global_step_val)
        else:
            _,_,loss_val=sess.run([train_op, metrics_update_op, loss])
        progress_bar.set_postfix(loss=round(loss_val,2))
            
    
    logging.info("- Train metrics: {}".format(loss_val))



def train_and_save(model_spec, model_dir, params, restore_weights = None):
    '''Train the model and save the weights
    Args:
        model_spec: (dict) contains all graph operations for training the model
        model_dir: (string) directory path to store weights and summaries
        params: (dict) hyperparameters of the model
        restore_weights: (string) directory path to restore weights from

    Returns:
        None
    '''

    last_saver = tf.train.Saver()
    
    with tf.Session() as sess:
        begin_epoch = 0
        sess.run(model_spec['variables_init_op'])
        if restore_weights is not None:
            logging.info('Restoring weights from {}'.format(restore_weights))
            latest_ckpt = tf.train.latest_checkpoint(os.path.join(restore_weights, 'weights'))
            begin_epoch = int(latest_ckpt.split('-')[-1])
            last_saver.restore(sess, latest_ckpt)
            

        writer = tf.summary.FileWriter(os.path.join(model_dir, 'summary'), sess.graph)
        
        
        for epoch in range(begin_epoch, begin_epoch + params['num_epochs']):
            logging.info('Epoch {}/{}'.format(epoch+1, begin_epoch + params['num_epochs']))
            num_steps = (params['train_size'] + params['batch_size'] - 1)//params['batch_size']
            train_sess(sess, model_spec, num_steps, writer, params)
            
            save_path = os.path.join(model_dir, 'weights', 'epoch')
            last_saver.save(sess, save_path, global_step = epoch+1)