import os
from bs4 import BeautifulSoup
import requests,urllib
import pandas as pd 

df=pd.read_csv("species.txt",sep='\t')
df.dropna(subset=['N/A'],how='all',inplace=True)
df.columns =['Name','Formula','CAS']
df.CAS=df.CAS.str.replace('-','')

# ls=[]
# for root, dirs, files in os.walk('./IR/'):
# 	for name in files:
# 		ls.append(name[:-6])

# df=df[~df.CAS.isin(ls)]
# print df.shape

url = "https://webbook.nist.gov/cgi/cbook.cgi"
# spectype = ['Mass','IR']
# response = requests.get(url, params={'JCAMP': 'C'+df.CAS.iloc[0], 'Type': 'Mass', 'Index': 0})

# print shap
# name=df.iloc[df.index.get_loc(df[df['CAS']=='79072'].index[0])].name
# print df.loc[29952:]['CAS']
for i in df.loc[:,'CAS']:
	for j in range(5):
		response = requests.get(url, params={'JCAMP': i, 'Type': 'IR', 'Index': j})
		if response.text == '##TITLE=Spectrum not found.\n##END=\n':
			break
		with open('./IR/'+str(i)+'d'+str(j)+'.jdx', 'w') as file:
			file.write(response.content)
		print(i,j)


# f=open("4447216.jdx")
# val = f.readline()
# if(val=="##TITLE=Spectrum not found.\n"):
# 	os.remove("4447216.jdx")