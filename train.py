import os
import logging
import argparse
import json

import tensorflow as tf

from model.utils import set_logger, train_test_generator
from model.input_fn import input_fn
from model.ae_model_fn import  ae_model_fn
from model.mlp_model_fn import mlp_model_fn
from model.train_fn import train_and_save
from model.embeddings_fn import embeddings
from prepare_load_dataset import load_dataset

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

logging.info('Load the dataset from data_dir')
X, y = load_dataset(args.data_dir, include_mass= True)


#Train and test generator for every fold
data_generator = train_test_generator(X, y, params['n_splits'])


for cv, (train_data, test_data) in enumerate(data_generator):
    logging.info('Starting {} fold'.format(cv))
    
    
    if params['train_ae']:
        tf.reset_default_graph()
        logging.info('Training autoencoder to compute embeddings')

        logging.info('Creating the inputs for the model')
        train_inputs = input_fn(True, train_data, params)
        eval_inputs = input_fn(False, test_data, params)

        logging.info('Building the model')
        train_model = ae_model_fn(True, train_inputs, params)
        eval_model = ae_model_fn(False, eval_inputs, params)


        logging.info('Start training {} epochs'.format(params['num_epochs']))
        model_dir = os.path.join(args.model_dir, str(cv), 'ae')
        train_and_save(train_model, eval_model, model_dir, params, restore_weights = args.restore_ae_from)

        #Update spectra data with embeddings computed from the model
        logging.info('Compute embeddings of the spectra data')
        train_data, test_data = embeddings(train_model, eval_model, model_dir, params, 'best_weights')

    tf.reset_default_graph()
    logging.info('Training MLP model')

    logging.info('Creating the inputs for the model')
    train_inputs = input_fn(True, train_data, params)
    eval_inputs = input_fn(False, test_data, params)

    logging.info('Building the model')
    train_model = ae_model_fn(True, train_inputs, params)
    eval_model = ae_model_fn(False, eval_inputs, params)

    logging.info('Start training {} epochs'.format(params['num_epochs']))
    model_dir = os.path.join(args.model_dir, 'cv_' + str(cv), 'mlp')
    train_and_save(train_model, eval_model, model_dir, params, restore_weights = args.restore_mlp_from)

