# CANDIY-spectrum

Human analyis of chemical spectra such as Mass Spectra (MS), Infra-Red Specta (FTIR), and Nuclear Magnetic Resonance is both time consuming and potentially inaccurate. This project aims to develop a set of methodologies incorporating these spectra for the prediction of chemical functional groups and structures.

This project is a stub, but we hope that it will spur development of machine learning methods for the analysis of chemical spectra.


### Required Packages
1) Numpy  
2) Matplotlib
3) pandas
4) jcamp
5) keras
6) tensorflow 
7) sklearn

### Data Scraping
IR and MS spectra were downloaded from NIST website. https://webbook.nist.gov/chemistry/. 
Scraping can be done through replacing the correct CAS number in the placeholder. https://webbook.nist.gov/cgi/cbook.cgi?ID="insert_cas"&Units=SI

(Or)

Use the scrap.py file to get the data. you can download all species names available in NIST from this link https://webbook.nist.gov/chemistry/download/

### Usage
1) Use jdxread.py to load jdx files (Point to the right folder) and create pickled files.
2) Use dataread.py to load pickled files and interpolate missing values
3) Use nnet.py or nnet_ae.py for loading the csv data and building the model.
