# -*- coding: utf-8 -*-
"""

@author: Rahul
"""

#Import the python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

#Get the dataset
dataset=pd.read_csv('timestamp.csv',skiprows=6,skipfooter=9,engine='python')
dataset['Adjustments']=pd.to_datetime(dataset['Adjustments'])+MonthEnd(1)

#Set the index
dataset=dataset.set_index('Adjustments')

#plot the data
dataset.plot()

#Split the data
split_date=pd.Timestamp('01-01-2011')

#Divide into traning and testing data
train=dataset.loc[:split_date,['Unadjusted']]
test=dataset.loc[split_date:,['Unadjusted']]

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_sc=sc.fit_transform(train)
test_sc=sc.transform(test)

#Divide data into traning and testing
train_sc[:4]
x_train=train_sc[:-1]
y_train=train_sc[1:]
x_test=test_sc[:-1]
y_test=test_sc[1:]

#Get the keras libraries
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as k
from keras.layers import LSTM
from keras.callbacks import EarlyStopping 

#Reshape the data
x_train[:,None].shape
x_train_t=x_train[:,None]
x_test_t=x_test[:,None]

#initialize the  RNN model
k.clear_session()
model=Sequential()
model.add(LSTM(6,input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

#Fitting  the Model into training data
model.fit(x_train_t,y_train,epochs=20,batch_size=1,verbose=1)

#Prediction on Test data
y_pred=model.predict(x_test_t)

#Display the results
plt.plot(y_test)
plt.plot(y_pred)











