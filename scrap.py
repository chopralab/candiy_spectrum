import os
from bs4 import BeautifulSoup as bs
import requests, urllib
import argparse
import logging
import pandas as pd 

from model.utils import set_logger


def scrap_data(cas_ls, params, data_dir):
	'''Collect data from NIST database and store them in jdx format.

    Args:
        cas_ls: (list) CAS ids to download data for
		params: (dict) queries to be added to url
		data_dir: (string) path to store the data

    Returns:
        None
    '''
	nist_url = "https://webbook.nist.gov/cgi/cbook.cgi"

	#Create directory for the relevant spetra 
	spectra_path = os.path.join(data_dir, params['Type'].lower(), '')
	if not os.path.exists(spectra_path):
		os.makedirs(spectra_path)

	num_created = 0
	for cas_id in cas_ls:
		params['JCAMP'] = 'C' + cas_id
		response = requests.get(nist_url, params=params)

		if response.text == '##TITLE=Spectrum not found.\n##END=\n':
			continue
		num_created+=1
		logging.info('Creating {} spectra for id: {}. Total spectra created {}'.format(params['Type'].lower(), cas_id, num_created))
		with open(spectra_path +cas_id +'.jdx', 'wb') as data:
			data.write(response.content)
			
	



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= './data',\
     help = "Directory path to store scrapped data")
parser.add_argument('--cas_smiles_list', default= 'species.txt',\
    help = "File containing CAS number and smiles of molecules")
parser.add_argument('--scrap_IR', default= True,\
    help = "Whether to download IR or not")
parser.add_argument('--scrap_MS', default= True,\
    help = "Whether to download MS or not")

args = parser.parse_args()

#Check if file containing CAS and smiles exist
assert os.path.isfile(args.cas_smiles_list),"No file named {} exists".format(args.cas_smiles_list)

#Create data directory to store logs and spectra
data_dir = args.data_dir
if not os.path.exists(data_dir):
	os.makedirs(data_dir)

set_logger(data_dir, 'scrap.log')

#Obtain CAS ids used for downloading the content from NIST
logging.info('Loading CAS file')
cas_smiles_df = pd.read_csv(args.cas_smiles_list, sep='\t', names = ['name', 'smiles', 'cas'], header = 0)
cas_smiles_df.dropna(subset=['cas'], inplace=True)
cas_smiles_df.cas = cas_smiles_df.cas.str.replace('-', '')

cas_ids = list(cas_smiles_df.cas)




logging.info('Scrap Mass spectra')
if args.scrap_MS:
	params = params={'JCAMP': '',  'Index': 0, 'Type': 'Mass'}
	scrap_data(cas_ids, params, data_dir)

logging.info('Scrap IR spectra')
if args.scrap_IR:
	params={'JCAMP': '', 'Type': 'IR', 'Index': 0}	
	scrap_data(cas_ids, params, data_dir)