import tensorflow as tf 


def input_fn(is_training, data, params):
    ''' Input function for spectra dataset

    Args:
        is_training: (bool) Whether it is training or not
        data: (tuple) containing (spectra, target) arrays. 
        params: (dict) Hyperparameters of the model

    Returns:
        inputs: (dict) Contains the iterator and data to be fed to the model

    '''

    #Shuffle training dataset
    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices(data)\
                    .shuffle(len(data))\
                    .batch(params['batch_size'])\
                    .prefetch(1)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)\
                    .batch(params['batch_size'])\
                    .prefetch(1)

    #Create initializable iterator to re-feed data after every epoch
    iterator = dataset.make_initializable_iterator()
    spectra_data, target = iterator.get_next()
    
    iterator_initializer_op = iterator.initializer
    
    inputs = {'spectra_data' : spectra_data, 'target' : target, 'iterator_initializer': iterator_initializer_op}
        
    return inputs
        