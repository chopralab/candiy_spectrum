import os
import logging

def set_logger(model_dir):
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
        fh = logging.FileHandler(os.path.join(model_dir, 'generation.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        #Stream handler with info level written to terminal
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    
    return logger
    
    
    
if __name__ == '__main__':
    model_dir = './mcts_logs/'
    set_logger(model_dir)

    