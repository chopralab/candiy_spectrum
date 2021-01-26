import os
import logging
import argparse
import sys

import pandas as pd 
import numpy as np
from jcamp import jcamp_read
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from model.utils import set_logger


# Initialize all constants necessary for standardizing the spectra
min_ir = 399
max_ir = 4001
step_ir = 3.25

min_mass = 1 
max_mass = 650
step_mass = 1

eps = 1e-4

# Create dictionary of functional group names and their corresponding smarts string
func_grp_smarts = {'alkane':'[CX4;H0,H1,H2,H4]','methyl':'[CH3]','alkene':'[CX3]=[CX3]','alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]','amines':'[NX3;H2,H1;!$(NC=O)]', 'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]','acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'}



                
def JCAMP_reader(filename):
    '''Overload function in jcamp to use latin-1 encoding instead of utf-8

    Args:
        filename: (string) jdx file containing spectra

    Returns:
        data: (dict) contains parsed information from file
    '''
    with open(filename, 'r', encoding = 'latin-1') as filehandle:
        data = jcamp_read(filehandle)
    data['filename'] = filename
    return data


def check_spectra_prop(mol_dict):
    '''Check if IR spectra satisfies certain conditions to be included in dataset

    Args:
        mol_dict: (dict) contains information about the spectra data

    Returns:
        _: (bool) whether spectra meets conditions
    '''
    cond1 = mol_dict.get('state', r'N\A').lower() == 'gas'
    cond2 = mol_dict.get('xunits', r'N\A').lower() != 'micrometers'
    cond3 = mol_dict.get('yunits', r'N\A').lower() == 'absorbance'
    
    return all((cond1, cond2, cond3))

def add_spectra_to_df(spectra_df, file_path, bins, is_mass = False):
    '''Add a spectra from filepath to the dataframe after standardizing

    Args:
        spectra_df: (pd.DataFrame) contains standardized spectra
        file_path: (string) path containing jdx file 
        bins: (np.array) used for standardizing
        is_mass: (bool) whether data being parsed is Mass or IR

    Returns:
        spectra_df: (pd.DataFrame) contains new spectrum aded to dataframe
    '''

    mol_dict = JCAMP_reader(file_path)

    #if conditions are not met, don't add the data
    if not is_mass and not check_spectra_prop(mol_dict):
        return spectra_df
    
    #Standardize the new spectrum and prepare for merging
    mol_id = mol_dict['cas registry no'].replace('-','')
    mol_xvalues = mol_dict['x']
    mol_yvalues = mol_dict['y']
    mol_df = pd.DataFrame(data = {mol_id : mol_yvalues}, index = mol_xvalues)
    mol_df.index = pd.cut(mol_df.index, bins = bins)
    mol_df = mol_df.groupby(level=0).agg('mean')

    logging.info('Adding spectra with id {} to dataframe'.format(mol_id))
    if spectra_df is None:
        spectra_df = mol_df
    else:
        spectra_df = pd.merge(spectra_df, mol_df, left_index = True, right_index = True, how='outer')
        
    return spectra_df

def save_spectra_to_csv(root, files, save_path, bins, is_mass = False):
    '''Save the spectra dataframe as csv to path

    Args:
        root: (string) path to spectra data
        files: (list) jdx files present in root
        save_path: (string) path to store csv file
        bins: (np.array) used for standardizing
        is_mass: (bool) whether data being parsed is Mass or IR

    Returns:
        None
    '''

    spectra_df = None
    for file_name in files:
        file_path = os.path.join(root,file_name)
        spectra_df = add_spectra_to_df(spectra_df, file_path\
                                                ,bins, is_mass)
    logging.info('Creating dataset in {}'.format(save_path))
    spectra_df.to_csv(save_path)


def identify_functional_groups(inchi):
    '''Identify the presence of functional groups present in molecule 
       denoted by inchi

    Args:
        root: (string) path to spectra data
        files: (list) jdx files present in root
        save_path: (string) path to store csv file
        bins: (np.array) used for standardizing
        is_mass: (bool) whether data being parsed is Mass or IR

    Returns:
        mol_func_groups: (list) contains binary values of functional groups presence
                          None if inchi to molecule conversion returns warning or error
    '''
    
    try:
        #Convert inchi to molecule
        mol = Chem.MolFromInchi(inchi, treatWarningAsError=True)   
        mol_func_grps = []

        #populate the list with binary values
        for _, func_struct in func_grp_structs.items():
            struct_matches = mol.GetSubstructMatches(func_struct)
            contains_func_grp = int(len(struct_matches)>0)
            mol_func_grps.append(contains_func_grp)
        return mol_func_grps
    except:

        return None
    
def save_target_to_csv(cas_inchi_df, save_path):
    '''Save the target dataframe as csv to path

    Args:
        cas_inchi_df: (pd.DataFrame) contains CAS and Inchi of molecules
        save_path: (string) path to store csv file

    Returns:
        None
    '''
    column_names = list(func_grp_structs.keys())    
    target_df = pd.DataFrame(index = cas_inchi_df.index, columns = column_names)

    #Iterate the rows, don't use df.apply since a list is being returned.
    for ind, (_, row) in enumerate(cas_inchi_df.iterrows()):
        target_df.iloc[ind, :] = identify_functional_groups(row['inchi'])
    

    target_df.dropna(inplace = True)
    target_df.to_csv(save_path)

def preprocess_spectra_df(spectra_df, is_mass = False, **kwargs):
    '''Preprocess the spectra dataframe by normalizing and interpolating

    Args:
        spectra_df: (pd.DataFrame) contains standardized spectra
        is_mass: (bool) whether data being parsed is Mass or IR
        kwargs: (dict) containing methods for interpolation

    Returns:
        spectra_df: (pd.DataFrame) contains processed spectra
    '''
    if is_mass:

        #Fill NaN with zero and remove m/z ratio where all values are zero
        spectra_df.fillna(0, inplace = True)
        spectra_df = spectra_df.loc[:,spectra_df.sum(axis=0)!=0]
        
    else:

        #Interpolate with linear or spline based on kwargs
        spectra_df.reset_index(inplace = True)
        spectra_df.iloc[:, 1:] = spectra_df.iloc[:,1:].interpolate(**kwargs,\
                                         limit_direction='both', axis = 0)
        spectra_df.set_index('index', inplace = True)

    #Normalize each spectra
    return spectra_df.div(spectra_df.max(axis=0), axis=1)
        


def load_dataset(data_dir, include_mass = True, **params):
    '''Load the spectra and target dataset for training

    Args:
        data_dir: (string) contains data path for csv file
        include_mass: (bool) whether to include mass spectra while training
        params: (dict) containing methods for interpolation

    Returns:
        X: (np.array) contains processed spectra values
        y: (np.array) contains target values of corresponding spectra
    '''

    #load and prepare IR data
    ir_path = os.path.join(data_dir, 'ir.csv')
    logging.info('Loading IR data from {}'.format(ir_path))
    ir_df = pd.read_csv(ir_path, index_col = 0)
    ir_df = preprocess_spectra_df(ir_df, is_mass = False, **params).T
    
    spectra_df = ir_df
    
    if include_mass:

        #Load and prepare mass data
        mass_path = os.path.join(data_dir, 'mass.csv')
        logging.info('Loading mass data from {}'.format(mass_path))
        mass_df = pd.read_csv(mass_path, index_col = 0).T
        mass_df = mass_df.loc[mass_df.index.isin(ir_df.index)]
        mass_df = preprocess_spectra_df(mass_df, is_mass = True)
        
#         mass_df = mass_df.reindex(ir_df.index)
#         spectra_df = pd.concat([spectra_df, mass_df], axis = 1)
#         spectra_df.dropna(inplace = True)

        #Merge mass data with IR
        spectra_df = pd.merge(spectra_df, mass_df, left_index = True, right_index = True, how = 'inner')
     
    #Prepare target data and rearrange to match the spectra
    spectra_df.index = spectra_df.index.astype('int')
    target_path = os.path.join(data_dir, 'target.csv')
    logging.info('Loading target data from {}'.format(target_path))
    target_df = pd.read_csv(target_path, index_col = 0, dtype = np.float64)

    fn_groups = target_df.shape[1]
    total_df = pd.merge(spectra_df, target_df, left_index = True, right_index = True, how = 'inner')
    
    return total_df.values[:, :-fn_groups], total_df.values[:, -fn_groups:], list(func_grp_smarts.keys())
    
    
if __name__ == '__main__':
    #Parsing the data from jdx and storing it in csv

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default= './data',\
        help = "Directory path containing scrapped data")
    parser.add_argument('--cas_list', default= 'species.txt',\
        help = "File containing CAS number and smiles of molecules")

    args = parser.parse_args()

    data_dir = args.data_dir
    set_logger(data_dir, 'prepare_data.log')


    # Create bins for IR and mass spectra
    logging.info('Creating bins for standardizing the spectra')
    ir_bins = np.arange(min_ir - eps, max_ir + eps, step_ir)
    mass_bins = np.arange(min_mass - eps, max_mass + eps, step_mass)

    # Compute structures of different molecular groups
    logging.info('Computing the structures of functional groups')
    func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts)\
                        for func_name, func_smarts in func_grp_smarts.items()}

    # Create and save csv files of spectra
    for root, dirs, files in os.walk(data_dir):
        if root == os.path.join(data_dir, 'ir'):
            logging.info('Starting to parse IR jdx files')
            ir_path = os.path.join(data_dir, 'ir.csv')
            save_spectra_to_csv(root, files, ir_path, ir_bins, False)

        if root == os.path.join(data_dir, 'mass'):
            logging.info('Starting to parse mass jdx files')
            mass_path = os.path.join(data_dir, 'mass.csv')
            save_spectra_to_csv(root, files, mass_path, mass_bins, True)
            
    #Load CAS data and merge with inchi
    logging.info('Loading CAS file from {}'.format(args.cas_list))
    cas_df = pd.read_csv(args.cas_list, sep='\t', header = 0, usecols = [1,2], names = ['formula','cas'])
    cas_df.dropna(subset=['cas'], inplace=True)
    cas_df.cas = cas_df.cas.str.replace('-', '')
    cas_df.set_index('cas', inplace = True)

    
    inchi_path = os.path.join(data_dir, 'inchi.txt')
    logging.info('Loading inchi file from {}'.format(inchi_path))
    inchi_df = pd.read_csv(inchi_path, sep='\t', header = 0, usecols = [0,1],\
                        names = ['cas','inchi'], dtype = str)
    inchi_df.dropna(inplace = True)
    inchi_df.set_index('cas', inplace = True)

    # Create and save csv of target 
    cas_inchi_df = pd.merge(cas_df, inchi_df, left_index = True, right_index = True, how = 'inner')
    target_path = os.path.join(data_dir, 'target.csv')
    logging.info('Creating target csv dataset in {}'.format(target_path))
    save_target_to_csv(cas_inchi_df, target_path)