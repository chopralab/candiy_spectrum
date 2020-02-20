import pandas as pd 
import keras
import tensorflow as tf 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD,Adam
from sklearn.metrics import accuracy_score,f1_score, roc_curve, roc_auc_score,make_scorer, precision_recall_curve,average_precision_score
import numpy as np 
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pickle
tf_config = tf.ConfigProto(allow_soft_placement=False)
# tf_config.gpu_options.allow_growth = True
from tensorflow import set_random_seed
from numpy.random import seed
seed(0)
set_random_seed(0)



df1 = pd.read_csv("mass.csv").T
X1 = df1.iloc[1:,:]
df2 = pd.read_csv("dataset.csv",index_col='name')
df2 = df2.reindex(map(int,X1.index))
X1 = X1.values.reshape(X1.shape)
X1=X1/np.reshape(X1.max(axis=1),(-1,1))
df3 = pd.read_csv("interpolateIR.csv").T
df3 = df3.reindex(df1.index)
X2= df3.iloc[1:,:]
X2 = X2.values.reshape(X2.shape)
X2=X2/np.reshape(X2.max(axis=1),(-1,1))
Y = df2.iloc[:,14:-4]
y = (Y>0).astype(int).values
X = np.concatenate([X1,X2],axis=1)



adam = keras.optimizers.Adam(lr=0.00015, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5	,
                              verbose=0, mode='auto')


def mlp_model():

	model = Sequential()
	model.add(Dense(200, input_dim=X.shape[1]))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.45))
	model.add(Dense(150))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.15))
	model.add(Dense(y.shape[1], activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	              optimizer=adam)
	return model

def ae_model():
	input_seq = Input((X.shape[1],))
	# h_1 = Dense(512, activation='relu')(input_seq)
	encode = Dense(256, activation='relu')(input_seq)
	# h_2 = Dense(512, activation='relu')(encode)
	output = Dense(X.shape[1], activation='sigmoid')(encode)
	model = Model(input_seq, output)
	model.compile(optimizer='adam', loss='mse')
	return model

model = ae_model()
model.fit(X, X, epochs=100, batch_size=50,verbose = 0)
encoder = Model(model.input,outputs= model.get_layer('dense_1').output)
transformed_X = encoder.predict(X)
# print (model.summary())
X = transformed_X

kfold = KFold(n_splits=5,shuffle=True,random_state=4)
ls=[]
trainls=[]
y = MultiLabelBinarizer().fit_transform(y)
print (y.shape)
print (X.shape)
for train_index, val_index in kfold.split(X, y):
	temp_ls = []
	traindat=[]
	model = mlp_model()
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	model.fit(X_train, y_train, validation_data=(X_val,y_val),callbacks=[earlystopping],epochs=100, batch_size=50,verbose = 0)
	preds = model.predict(X_val)
	trainpreds = model.predict(X_train)
	for i,col in enumerate(Y.columns):
		temp_ls.append(average_precision_score(y_val[:,i],preds[:,i]))
		traindat.append(average_precision_score(y_train[:,i],trainpreds[:,i]))
	ls.append(temp_ls)
	trainls.append(traindat)


train=np.array(trainls).mean(axis=0)
val = np.array(ls).mean(axis=0)
print(pd.DataFrame([train,val],columns= Y.columns,index = ['Train','val']).T)


#############TESTING################
# df1 = pd.read_csv("labmass.csv",index_col='x').T
# df1 = df1.divide(df1.max(axis=1),axis=0)
# df3 = pd.read_csv("labir.csv",header= 1,index_col=0)
# # df3 = pd.read_csv("labir.csv",index_col=0)
# # print(df3.head())
# # X_test = df3.iloc[1:,:].values.astype(float)

# # X_test=X_test/np.reshape(X_test.max(axis=1),(-1,1))

# # print (set(df1.index)-set(df3.index))

# df3 = df3.divide(df3.max(axis=1),axis=0)

# df1 = df1.merge(df3,right_index=True,left_index=True)
# print (df1.index)
# # print (df3.max(axis=1),df1.max(axis=1))
# X_test = df1.values
# # print (es)
# # X_test = df3.values

# model = mlp_model()
# model.fit(X, y,epochs=50, batch_size=90,verbose = 0)
# preds = model.predict(X_test)
# trainpreds = model.predict(X)
# # print(preds)
# # preds[preds>=thre_ir] = 1
# # preds[preds<thre_ir] = 0 
# # trainpreds[trainpreds>=thre] = 1
# # trainpreds[trainpreds<thre] = 0 
# temp_ls=[]
# for i,col in enumerate(Y.columns):
# 	temp_ls.append(average_precision_score(y[:,i],trainpreds[:,i]))
# print (pd.DataFrame(temp_ls,index= Y.columns))
# pd.DataFrame(preds,index=df1.index,columns= Y.columns).to_csv('test_predictions_IR+MS.csv')
# # print(f1_score(y,trainpreds,average = 'weighted'))
# # pd.DataFrame(preds,index=df3.index[1:],columns= Y.columns)