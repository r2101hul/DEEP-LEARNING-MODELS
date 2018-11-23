# -*- coding: utf-8 -*-
"""
Artificial Neural Network with PIMA Indian Diabetes DATASET

@author: Rahul
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset=pd.read_csv('diabetes.csv')
x=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split (x,y,test_size=0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Initialising the ANN
classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))

# Adding the second hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))

# Adding the second hidden layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training  Data
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred=classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




















