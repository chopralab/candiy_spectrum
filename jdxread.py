import os
from jcamp import JCAMP_reader
# from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt 
# import pandas as pd 
import pickle
from collections import Counter
import pandas as pd

dc={}
xu,yu=[],[]
for root, dirs, files in os.walk('./IR/'):
	for name in files:
		# temp_dc={}
		if name.endswith((".jdx")):

			jcamp_dict=JCAMP_reader(root+'/'+name)
			# xu.append(jcamp_dict['npoints'])
			dc[name[:-4]]= jcamp_dict
			# ls.append(jcamp_dict['x'].shape[0])
# print Counter(xu)
# del dc['']
with open('IR.pickle', 'wb') as handle:
    pickle.dump(dc, handle, protocol=pickle.HIGHEST_PROTOCOL)






###################### LAB data####################
# dc={}
# # xu,yu=[],[]
# for root, dirs, files in os.walk('./'):
# 	for name in files:
# 		# temp_dc={}
# 		if name.endswith((".JDX")):
# 			jcamp_dict=JCAMP_reader(root+'/'+name)
# 			# xu.append(jcamp_dict['npoints'])
# 			# print(jcamp_dict['xunits'],jcamp_dict['yunits'])
# 			if jcamp_dict['yunits'] == 'TRANSMITTANCE':
# 				print(name)
# 				jcamp_dict['y'] = 2 - np.log(100*jcamp_dict['y'])
# 			dc[name[0:2]+name[-6:-4]]= jcamp_dict
# 			# ls.append(jcamp_dict['x'].shape[0])

# with open('labir.pickle', 'wb') as handle:
#     pickle.dump(dc, handle, protocol=pickle.HIGHEST_PROTOCOL)


