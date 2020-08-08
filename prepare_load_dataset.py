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



min_ir = 399
max_ir = 4001
step_ir = 3.25

min_mass = 1 
max_mass = 650
step_mass = 1

eps = 1e-4

func_grp_smarts = {'alkane':'[CX4]','methyl':'[CH3]','alkene':'[CX3]=[CX3]','alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]','amines':'[NX3;H2,H1;!$(NC=O)]', 'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]','acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'}


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= './data',\
     help = "Directory path containing scrapped data")
parser.add_argument('--cas_list', default= 'species.txt',\
    help = "File containing CAS number and smiles of molecules")

args = parser.parse_args()


data_dir = args.data_dir
set_logger(data_dir, 'prepare_data.log')



ir_bins = np.arange(min_ir - eps, max_ir + eps, step_ir)
mass_bins = np.arange(min_mass - eps, max_mass + eps, step_mass)

func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts)\
                      for func_name, func_smarts in func_grp_smarts.items()}
                



def JCAMP_reader(filename):
    with open(filename, 'r', encoding = 'latin-1') as filehandle:
        data = jcamp_read(filehandle)
    data['filename'] = filename
    return data


def check_spectra_prop(mol_dict):
    cond1 = mol_dict.get('state', 'N\A').lower() == 'gas'
    cond2 = mol_dict.get('xunits', 'N\A').lower() != 'micrometers'
    cond3 = mol_dict.get('yunits', 'N\A').lower() == 'absorbance'
    
    return all((cond1, cond2, cond3))

def add_spectra_to_df(spectra_df, file_path, bins, is_mass = False):
    mol_dict = JCAMP_reader(file_path)
    if not is_mass and not check_spectra_prop(mol_dict):
        return spectra_df
    
    mol_id = mol_dict['cas registry no'].replace('-','')
    mol_xvalues = mol_dict['x']
    mol_yvalues = mol_dict['y']
    mol_df = pd.DataFrame(data = {mol_id : mol_yvalues}, index = mol_xvalues)
    mol_df.index = pd.cut(mol_df.index, bins = bins)
    mol_df = mol_df.groupby(level=0).agg('mean')

    if spectra_df is None:
        spectra_df = mol_df
    else:
        spectra_df = pd.merge(spectra_df, mol_df, left_index = True, right_index = True, how='outer')
        
    return spectra_df

def save_spectra_to_csv(root, files, save_path, bins, is_mass = False):
    spectra_df = None
    for file_name in files:
        file_path = os.path.join(root,file_name)
        spectra_df = add_spectra_to_df(spectra_df, file_path\
                                                ,bins, is_mass)
    spectra_df.to_csv(save_path)


def identify_functional_groups(inchi):
    
    try:
        mol = Chem.MolFromInchi(inchi, treatWarningAsError=True)   
        mol_func_grps = []
        for _, func_struct in func_grp_structs.items():
            struct_matches = mol.GetSubstructMatches(func_struct)
            contains_func_grp = int(len(struct_matches)>0)
            mol_func_grps.append(contains_func_grp)
        return mol_func_grps
    except:

        return None
    
def save_target_to_csv(cas_inchi_df, save_path):
    column_names = list(func_grp_structs.keys())    
    target_df = pd.DataFrame(index = cas_inchi_df.index, columns = column_names)
    for ind, (_, row) in enumerate(cas_inchi_df.iterrows()):
        target_df.iloc[ind, :] = identify_functional_groups(row['inchi'])
    

    target_df.dropna(inplace = True)
    target_df.to_csv(save_path)
    
    
if __name__ == '__main__':
    for root, dirs, files in os.walk(data_dir):
        if root == os.path.join(data_dir, 'ir'):
            ir_path = os.path.join(data_dir, 'ir.csv')
            save_spectra_to_csv(root, files, ir_path, ir_bins, False)

        if root == os.path.join(data_dir, 'mass'):
            mass_path = os.path.join(data_dir, 'mass.csv')
            save_spectra_to_csv(root, files, mass_path, mass_bins, True)
            

    cas_df = pd.read_csv(args.cas_list, sep='\t', header = 0, usecols = [1,2], names = ['formula','cas'])
    cas_df.dropna(subset=['cas'], inplace=True)
    cas_df.cas = cas_df.cas.str.replace('-', '')
    cas_df.set_index('cas', inplace = True)

    inchi_path = os.path.join(data_dir, 'inchi.txt')
    inchi_df = pd.read_csv(inchi_path, sep='\t', header = 0, usecols = [0,1],\
                        names = ['cas','inchi'], dtype = str)
    inchi_df.dropna(inplace = True)
    inchi_df.set_index('cas', inplace = True)

    cas_inchi_df = pd.merge(cas_df, inchi_df, left_index = True, right_index = True, how = 'inner')
    target_path = os.path.join(data_dir, 'target.csv')
    save_target_to_csv(cas_inchi_df, target_path)