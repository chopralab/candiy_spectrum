import os
import logging
from tqdm import trange

import numpy as np
import tensorflow as tf

def embedding_sess(sess, model_spec, size, params):
    '''Compute embeddings of spectra data

    Args:
        sess: (tf.Session) indicates current session
        model_spec: (dict) contains graph operations for computing embeddings
        size: (int) dataset size 
        params: (dict) hyperparameters of the model


    Returns:
        data: (tuple) containing arrays of embedding and target
    
    '''
    
    #Compute dimension size to create data arrays
    target_dim = model_spec['target'].shape[-1]
    embed_dim = model_spec['embeddings'].shape[-1]

    #Initialize target and embedding array
    target_arr = np.zeros((size, target_dim)) 
    embed_arr = np.zeros((size, embed_dim))

    #Initiliaze the dataset iterator
    sess.run(model_spec['iterator_initializer'])
    

    batch_size = params['batch_size']
    #Compute number of batches
    num_steps = (size + batch_size- 1)//batch_size

    #Compute batch wise target and embeddings. Add it to data array. 
    progress_bar = trange(num_steps, position = 0)
    for step in progress_bar:
        embed_batch, target_batch = sess.run([model_spec['embeddings'], model_spec['target']])
        target_arr[step*batch_size: (step+1)* batch_size] = target_batch
        embed_arr[step*batch_size: (step+1)* batch_size] = embed_batch

    return embed_arr, target_arr

def embeddings(train_model_spec, eval_model_spec, model_dir, params, restore_weights):
    '''Compute the embeddings after restoring the weights

    Args:
        train_model_spec: (dict) contains operations for computing train embeddings
        eval_model_spec: (dict) contains operations for computing test embeddings
        model_dir: (string) directory path indicating model_dir
        params: (dict) hyperparameters of the model
        restore_weights: (string) directory path to restore weights from

    Returns:
        train_data: (tuple) containing arrays of X_train embedding and y_train
        test_data: (tuple) containing arrays of X_test embedding and y_test
    '''

    saver = tf.train.Saver()
  
    with tf.Session() as sess:
        #Restore weights from model_dir/restore_weights
        restore_dir = os.path.join(model_dir, restore_weights)
        logging.info('Restoring weights from {}'.format(restore_dir))
        latest_ckpt = tf.train.latest_checkpoint(restore_dir)
        saver.restore(sess, latest_ckpt)

        #Compute embeddings for train and test data
        train_data = embedding_sess(sess, train_model_spec, params['train_size'], params)
        test_data = embedding_sess(sess, eval_model_spec, params['eval_size'], params)

    return (train_data, test_data)