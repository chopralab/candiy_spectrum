import os
import logging

from sklearn.model_selection import KFold

def set_logger(model_dir, log_name):
    '''Set logger to write info to terminal and save in a file.

    Args:
        model_dir: (string) path to store the log file

    Returns:
        None
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Don't create redundant handlers everytime set_logger is called
    if not logger.handlers:

        #File handler with debug level stored in model_dir/generation.log
        fh = logging.FileHandler(os.path.join(model_dir, log_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        #Stream handler with info level written to terminal
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    
    return logger

def train_test_generator(X, y, n_splits):
    '''
    Create a generator to return next train and test data split when called

    Args:
        X: (np.array) of dimension [num_samples x features]
        y: (np.array) of dimension [num_samples x target groups]
        n_splits: (int) Number of cross validation folds
    
    Returns:
        (X_train, y_train): (tuple) of np.arrays containing single fold of train data
        (X_test, y_test): (tuple) of np.arrays containing single fold of test data
    '''

    kfold = KFold(n_splits=5,shuffle=True,random_state=4)

    for train_index, val_index in kfold.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        yield (X_train, y_train), (X_val, y_val)


    