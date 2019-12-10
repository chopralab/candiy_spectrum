import os
from jcamp import JCAMP_reader
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle

with open('./Formatted_data/IR.pickle', 'rb') as handle:
    dc = pickle.load(handle)

datals=[]
ls=[]
for i in dc.keys():
		df = pd.DataFrame(list(zip(dc[i]['x'],dc[i]['y'])),columns= ['x',i])
		df=df.groupby('x').aggregate(np.mean)
		df.reset_index(inplace=True)
		datals.append(df)
		ls.append([np.min(np.diff(df.x)),np.min(df.x),np.max(df.x)])
arr = np.array(ls)
bins = np.arange(np.min(arr[:,1])-0.1,np.max(arr[:,2])+0.1,np.mean(arr[:,0]))

datals[0]['x'] = pd.cut(datals[0]['x'],bins)
df=datals[0].groupby('x').aggregate(np.mean)
df.reset_index(inplace=True)

for i,df_ in enumerate(datals[1:]):
	df_['x'] = pd.cut(df_['x'],bins)
	df_=df_.groupby('x').aggregate(np.mean)
	df_.reset_index(inplace=True)
	df = df.merge(df_, on='x',how='outer')
df.iloc[:,1:] = df.iloc[:,1:].interpolate(limit_direction='both',axis=0)
df.to_csv("interpolateIR.csv",index=False)





			

