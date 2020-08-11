# CANDIY-spectrum

Human analyis of chemical spectra such as Mass Spectra (MS), Infra-Red Specta (FTIR), and Nuclear Magnetic Resonance is both time consuming and potentially inaccurate. This project aims to develop a set of methodologies incorporating these spectra for the prediction of chemical functional groups and structures.

This project is a stub, but we hope that it will spur development of machine learning methods for the analysis of chemical spectra.


## Required Packages
1) Numpy==1.18.5
2) Rdkit==2020.03.1
3) Pandas==1.0.5
4) Jcamp
5) Tensorflow==1.15.0 
6) Matplotlib==3.1.1
7) Sklearn==0.22.1

## Scraping
### Manual Scraping
IR and MS spectra were downloaded from NIST website. https://webbook.nist.gov/chemistry/. 
Scraping can be done through replacing the correct CAS number in the placeholder. https://webbook.nist.gov/cgi/cbook.cgi?ID="insert_cas"&Units=SI and downloading the required spectra. 

(Or)
### Automatic Scraping
Download all the species name available in NIST from this link https://webbook.nist.gov/chemistry/download/. Change path of cas_list to where the species name file is stored.
```
python scrap.py --save_dir='./data/' --cas_list='species.txt' --scrap_IR=true --scrap_MS=true --scrap_InChi=true
```

## Prepare dataset
Parse all jdx files of IR and Mass spectra to standardize and store in a csv format. Also, parse inchi.txt to create target csv indicating presence of functional groups

```
python prepare_load_dataset.py --data_dir='./data/' --cas_list='species.txt'
```

## Train the model
Run train.py to train the model. The model directory should contain params.json file listing all hyperparameters used for building the model. Optionally weights can be restored from pretrained model. 

```
python train.py --model_dir=./experiments/mlp_model --data_dir=./data/ 
```
