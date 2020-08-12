import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,f1_score, accuracy_score


def compute_thresholds(data_ls):
    data_arr = np.concatenate(data_ls, axis = 1)
    data_target = data_arr[1]
    data_probs = data_arr[0]
    
    num_func_groups = data_target.shape[1]
    thresholds = np.zeros((1, num_func_groups))
    eps = 1e-7 
    
    for i in range(num_func_groups):
        pre,rec,thre=precision_recall_curve(data_target[:,i],data_probs[:, i])
        f1 = 2*pre*rec/(pre+rec+eps)
        max_ind = np.argmax(f1)
        thresholds[0,i] = thre[max_ind]
    return thresholds

def compute_metrics(data_ls, thresholds, func_names):
    num_folds = len(data_ls)
    num_groups = thresholds.shape[1]
    mol_f1 = np.zeros((num_folds, num_groups))
    mol_perf = np.zeros((num_folds, 1))  
                
    for ind, data_fold in enumerate(data_ls):
        fold_target = data_fold[1]
        fold_preds = (data_fold[0]>thresholds).astype('int')
        mol_f1[ind,:] = f1_score(fold_target, fold_preds, average = None)
        mol_perf[ind,:] = accuracy_score(fold_target, fold_preds)
        
    overall_perf = np.array([[np.mean(mol_perf), np.std(mol_perf)]])
    overall_f1 = np.array([np.mean(mol_f1, axis = 0), np.std(mol_f1, axis = 0)])
    
    mol_f1_df = pd.DataFrame(overall_f1, index=['mean', 'std'], columns = func_names).T
    mol_perf_df = pd.DataFrame(overall_perf, columns=['mean', 'std'])
    
    return mol_perf_df, mol_f1_df 


def store_results(train_predictions, test_predictions, func_group_names, save_path):
    thresholds = compute_thresholds(test_predictions)
    train_perf_df,train_f1_df = compute_metrics(train_predictions, thresholds, func_group_names)
    test_perf_df,test_f1_df = compute_metrics(test_predictions, thresholds, func_group_names)
    
    f1_df = pd.concat([train_f1_df, test_f1_df], keys = ['Train', 'Val'], axis = 1)
    perf_df = pd.concat([train_perf_df, test_perf_df], axis = 0)
    perf_df.index = ['Train', 'Val']
    
    f1_df.to_csv(os.path.join(save_path, 'mol_f1.csv'))
    perf_df.to_csv(os.path.join(save_path, 'mol_perf.csv'))

    