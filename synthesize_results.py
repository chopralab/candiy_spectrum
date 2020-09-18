import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,f1_score, accuracy_score


def compute_thresholds(data_ls):
    '''Compete dynamic thresholds for every functional group using val data

    Args:
        data_ls: (list) containing val predictions and target of all folds

    Returns:
        thresholds: (np.array) containing the thresholds of groups
    '''
    logging.info('Computing Thresholds')
    #Combine all test data into singel array
    data_arr = np.concatenate(data_ls, axis = 1)
    data_target = data_arr[1]
    data_probs = data_arr[0]
    
    num_func_groups = data_target.shape[1]
    thresholds = np.zeros((1, num_func_groups))
    eps = 1e-7 
    
    #Find threshold resulting in maximum f1 score for each functional group
    for i in range(num_func_groups):
        pre,rec,thre=precision_recall_curve(data_target[:,i],data_probs[:, i])
        f1 = 2*pre*rec/(pre+rec+eps)
        max_ind = np.argmax(f1)
        thresholds[0,i] = thre[max_ind]
    return thresholds

def compute_metrics(data_ls, thresholds, func_names):
    '''Compete metrics for every fold of train and val data

    Args:
        data_ls: (list) containing predictions and target of all folds
        thresholds: (np.array) of every functional group
        func_names: (list) used as part of target

    Returns:
        mol_score_df: (pd.DataFrame) contains mean and std of mol. scores
        func_f1_df: (pd.DataFrame) contains mean and std of func. f1 score
    '''

    logging.info('Computing func_f1, mol_f1 and mol_perfection metrics')
    num_folds = len(data_ls)
    num_groups = thresholds.shape[1]
    func_f1= np.zeros((num_folds, num_groups))
    mol_score = np.zeros((num_folds, 2))  
                
    #Using thresholds find func_f1, mol_f1 and mol_perfection for all folds
    for ind, data_fold in enumerate(data_ls):
        fold_target = data_fold[1]
        fold_preds = (data_fold[0]>thresholds).astype('int')
        func_f1[ind,:] = f1_score(fold_target, fold_preds, average = None)
        mol_score[ind,1] = f1_score(fold_target, fold_preds, average = 'samples')
        mol_score[ind,0] = accuracy_score(fold_target, fold_preds)
        
    overall_score = np.array([np.mean(mol_score, axis = 0), np.std(mol_score,axis = 0)])
    overall_f1 = np.array([np.mean(func_f1, axis = 0), np.std(func_f1, axis = 0)])
    
    print (overall_f1.shape, overall_score.shape)
    #Create a dataframe with the results
    func_f1_df = pd.DataFrame(overall_f1, columns = func_names, index=['mean', 'std']).T
    mol_score_df = pd.DataFrame(overall_score, columns = ['mol. perfection', 'mol. f1'], index=['mean', 'std']).T
    
    return mol_score_df, func_f1_df 


def store_results(train_predictions, test_predictions, func_group_names, save_path):
    '''Store results in a csv file

    Args:
        data_ls: (list) containing predictions and target of all folds
        thresholds: (np.array) of every functional group
        func_names: (list) used as part of target

    Returns:
        None
    '''
    thresholds = compute_thresholds(test_predictions)
    train_score_df,train_f1_df = compute_metrics(train_predictions, thresholds, func_group_names)
    test_score_df,test_f1_df = compute_metrics(test_predictions, thresholds, func_group_names)
    
    f1_df = pd.concat([train_f1_df, test_f1_df], keys = ['Train', 'Val'], axis = 1)
    perf_df = pd.concat([train_score_df, test_score_df], keys = ['Train', 'Val'], axis = 1)
    
    logging.info('Storing results in {}'.format(save_path))
    f1_df.to_csv(os.path.join(save_path, 'func_f1.csv'))
    perf_df.to_csv(os.path.join(save_path, 'mol_score.csv'))

    