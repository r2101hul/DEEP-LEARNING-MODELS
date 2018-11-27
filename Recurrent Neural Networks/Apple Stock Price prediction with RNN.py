# -*- coding: utf-8 -*-
"""


@author: Rahul
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing The Dataset
dataset_train=pd.read_csv('appl_Stock.csv')
training_set=dataset_train.iloc[:,1:2].values

#Scaling The data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#Get the training set
x_train=[]
y_train=[]
for i in range(60,1761):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)    
    
#Reshape the data in 3D
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1] ,1))

#Importing the keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialize the Model
regressor=Sequential()
#Adding First Layer
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding the second layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the third layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Output layer
regressor.add(Dense(units=1))

#Compile the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the Model Training Data
regressor.fit(x_train,y_train,epochs=100,batch_size=32)

#Get the test data
dataset_test=pd.read_csv('appl_jan.csv')
real_stock_price=dataset_test.iloc[:,1:2].values
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)


inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

x_test=[]
for i in range (60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)

#Reshape the Test  data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Predict the stock_price
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visulisations of the  results   
plt.plot(real_stock_price,color='red',label='Real')
plt.plot(predicted_stock_price,color='green',label='predicted' )
plt.xlabel('Time')
plt.ylabel('Apple stock price ')
plt.legend()
plt.show()

    









