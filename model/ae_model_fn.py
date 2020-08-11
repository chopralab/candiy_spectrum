import tensorflow as tf 

def build_ae_model(is_training, inputs, params):
    '''Build forward model and reconstruct the spectra

    Args:
        is_training: (tf.placeholder) indicates training or evaluation
        inputs: (dict) contains tensors of inputs fed to the graph
        params: (dict) hyperparameters of the model

    Returns:
        emb_layer: (tf.tensor) Embeddings computed by the encoder
        output: (tf.tensor) Reconstructed spectra
    
    '''
    
    #Read all hyperparameters
    num_ae_layers = params['num_fc_layers']
    ae_hidden_units = params['fc_hidden_units']
    activation = params['activation']
    is_denoising = params.get('is_denoising', False)
    denoise_prob = params.get('denoise_inputs', 0.05)
    hidden_layer = inputs 

    # Randomly flip inputs to 0 with the probability of denoise_prob
    if is_denoising:
        input_shape = tf.shape(inputs)
        hidden_layer *= tf.where(tf.random_uniform(input_shape) > denoise_prob, tf.ones(input_shape)\
                            , tf.zeros(input_shape))

    #Construct hidden layers of the encoder
    for layer in range(num_ae_layers):
        with tf.variable_scope('enc_{}'.format(layer+1)):
            hidden_layer = tf.layers.dense(hidden_layer, ae_hidden_units[layer], eval(activation))
            # batch_norm_layer = tf.layers.batch_normalization(hidden_layer, training = is_training)
            # activation_layer = eval(activation)(batch_norm_layer)
            # dropout_layer = tf.layers.dropout(activation_layer, rate = dropout_probs[layer],training = is_training)

            
            

    emb_layer = hidden_layer

    #Construct hidden layers of the decoder
    for layer in range(num_ae_layers-2, -1, -1):
        with tf.variable_scope('dec_{}'.format(layer+1)):
            hidden_layer = tf.layers.dense(hidden_layer, ae_hidden_units[layer], eval(activation))
            # batch_norm_layer = tf.layers.batch_normalization(hidden_layer, training = is_training)
            # activation_layer = eval(activation)(batch_norm_layer)
            # dropout_layer = tf.layers.dropout(activation_layer, rate = dropout_probs[layer],training = is_training)
    
    #Compute reconstructed spectra (use sigmoid as activation to get [0,1] range like input)
    with tf.variable_scope('dec_{}'.format(layer+1)):
        output = tf.layers.dense(hidden_layer, inputs.shape[-1], 'sigmoid')
    
    return emb_layer, output


def ae_model_fn(is_training, inputs, params):

    '''Define graph operations for training and evaluating
    
    Args:
        is_training: (bool) indicates training or evaluation
        inputs: (dict) contains tensors of inputs and labels fed to the graph
        params: (dict) hyperparameters of the model

    Returns:
        model_spec: (dict) Contains the operations needed for training and evaluating the model
    
    '''

    
    spectra_data = inputs['spectra_data']
    is_train_ph = tf.placeholder_with_default(is_training, shape=()) #Define a placeholder for setting mode during evaluation

    #Compute embeddings and reconstructed data
    with tf.variable_scope('model', reuse = not is_training):
        embeddings, spectra_recon = build_ae_model(is_train_ph, spectra_data, params)
        
    #Mean squared loss between input and reconstructed spectra
    loss = tf.losses.mean_squared_error(spectra_data, spectra_recon)




    if is_training:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        global_step = tf.train.get_or_create_global_step()
        
        #Perform update_op to update moving mean and variance before minimizing the loss
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op = optimizer.minimize(loss, global_step = global_step)
    
    
    with tf.variable_scope('metrics'):
        metrics = {'loss' : tf.metrics.mean(loss)}
        
    
        
    #Group all metrics update ops 
    metrics_update_op = tf.group(*[metric[1] for _, metric in metrics.items()])
        
    #Collect all metrics variables to initialize before every epoch
    metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_initializer_op = tf.variables_initializer(metrics_variables)
        
    tf.summary.scalar('loss', loss)
        
    model_spec = inputs
    model_spec['loss'] = loss
    model_spec['embeddings'] = embeddings
    model_spec['metrics'] = metrics
    model_spec['metric_initializer_op'] = metrics_initializer_op
    model_spec['metrics_update_op'] = metrics_update_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['variables_init_op'] = tf.global_variables_initializer()
    model_spec['train_ph'] = is_train_ph
    
    
    if is_training:
        model_spec['train_op'] = train_op
    
    return model_spec