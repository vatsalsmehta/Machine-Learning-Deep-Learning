# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:19:28 2020

@author: vatsal
"""


import numpy as np
import pandas as pd


df=pd.read_csv('datamodel2.csv')

print("\n")

print(df.shape)

print("\n")

print(df.columns)

print("\n")

print(df.dtypes)

print("\n")

print(df.corr())

print("\n")

print(df.isnull().sum())

"""Segregating data in two"""

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


"""Encoding Categorical data"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print('X')


"""Splitting data into training and testing"""
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

"""calling Multiple Linear Regression Model"""
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train) 

y_pred=lr.predict(X_test)

print("\n")
print("Predicted values         Actual Values")
for i in range(0,12):
    print(y_pred[i],"    ",y_test[i])


#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))






















