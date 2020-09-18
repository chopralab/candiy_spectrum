import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import argparse
import json

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from model.utils import set_logger, train_test_generator
from model.input_fn import input_fn
from model.ae_model_fn import  ae_model_fn
from model.mlp_model_fn import mlp_model_fn
from model.train_fn import train_and_save
from model.evaluate_fn import evaluate_and_predict
from prepare_load_dataset import load_dataset
from synthesize_results import store_results

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default= './experiments/ae_mlp_model',\
     help = "Directory path containing params.json and to store summary and weights")
parser.add_argument('--data_dir', default= './data',\
    help = "Directory path containing IR and MS spectra data")
parser.add_argument('--restore_ae_from', default= None,\
    help = "Restore AE weights before training the model")
parser.add_argument('--restore_mlp_from', default= None,\
    help = "Restore MLP weights before training the model")

args = parser.parse_args()

#Model directory should contain params.json file listing all hyperparameters
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path),"No params.json found at {} path".format(args.model_dir)

with open(json_path) as json_data:
    params = json.load(json_data)

set_logger(args.model_dir, 'train.log')

logging.info('Load the dataset from {}'.format(args.data_dir))
X, y, func_names = load_dataset(args.data_dir, True, **params['preprocess'])


#Train and test generator for every fold
data_generator = train_test_generator(X, y, params['n_splits'])

train_predictions = []
test_predictions = []

for cv, (train_data, test_data) in enumerate(data_generator):
    logging.info('Starting fold {}'.format(cv+1))
    train_size = train_data[0].shape[0]
    eval_size = test_data[0].shape[0]
    
    if params['train_ae']:
        tf.reset_default_graph()
        logging.info('Training autoencoder to compute embeddings')

        ae_params = params['ae']
        ae_params['train_size'] = train_size
        ae_params['eval_size'] = eval_size

        logging.info('Creating the inputs for the model')
        train_inputs = input_fn(True, train_data, ae_params)
        eval_inputs = input_fn(False, test_data, ae_params)

        logging.info('Building the model')
        train_model = ae_model_fn(True, train_inputs, ae_params)
        eval_model = ae_model_fn(False, eval_inputs, ae_params)


        logging.info('Start training {} epochs'.format(params['ae']['num_epochs']))
        model_dir = os.path.join(args.model_dir, 'cv_' + str(cv+1), 'ae')
        train_and_save(train_model, eval_model, model_dir, ae_params, restore_weights = args.restore_ae_from)

        #Update spectra data with embeddings computed from the model
        logging.info('Compute embeddings of the spectra data')
        emb_params = {'restore_path' :os.path.join(model_dir,'best_weights'), 'params' :ae_params,\
                        'layer_name' :'embeddings', 'evaluate_model' :False}
        
        train_data = evaluate_and_predict(train_model, is_train_data = True, **emb_params)
        test_data = evaluate_and_predict(eval_model, is_train_data = False, **emb_params)

    tf.reset_default_graph()
    logging.info('Training MLP model')

    mlp_params = params['mlp']
    mlp_params['train_size'] = train_size
    mlp_params['eval_size'] = eval_size


    logging.info('Creating the inputs for the model')
    train_inputs = input_fn(True, train_data, mlp_params)
    eval_inputs = input_fn(False, test_data, mlp_params)

    logging.info('Building the model')
    train_model = mlp_model_fn(True, train_inputs, mlp_params)
    eval_model = mlp_model_fn(False, eval_inputs, mlp_params)

    logging.info('Start training {} epochs'.format(params['mlp']['num_epochs']))
    model_dir = os.path.join(args.model_dir, 'cv_' + str(cv+1), 'mlp')
    train_and_save(train_model, eval_model, model_dir, mlp_params, restore_weights = args.restore_mlp_from)

    logging.info('Compute prediction probabilities of the spectra data')
    pred_params = {'restore_path' :os.path.join(model_dir,'best_weights'), 'params' :mlp_params,\
                        'layer_name' :'pred_probs', 'evaluate_model' :False}
        
    #Compute prediction probabilites of the model to compute f1 and perfection rate
    train_data = evaluate_and_predict(train_model, is_train_data = True, **pred_params)
    test_data = evaluate_and_predict(eval_model, is_train_data = False, **pred_params)

    train_predictions.append(train_data)
    test_predictions.append(test_data)

#Compute and save the metrics
store_results(train_predictions, test_predictions, func_names, args.model_dir)

logging.info('Successfully Completed!!!!!')


    

