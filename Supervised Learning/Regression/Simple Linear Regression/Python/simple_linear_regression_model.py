# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:08:12 2020

@author: vatsal
"""

#import necessary librares

import numpy as np
import pandas as pd

"""importing dataset"""

df=pd.read_csv('datamodel.csv')

print(df.shape)
print("\n")

print(df.columns)
print("\n")

print(df.corr())
print("\n")

"""Checking for missing values"""

print(df.isnull().sum())
print("\n")

"""Segregating input and output values"""
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

"""Splitting for training and testing"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

"""Applying Linear Regression"""
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

print("\n")

"""y_pred conatins Predicted  Values"""
y_pred=regressor.predict(X_test)

print("\n")

"""Lets see predicted values vs Actual Values"""

print("Predicted Values(Salary)                    Actual Values(Salary)")
for i in range (0,len(X_test)):
    print(y_pred[i],"                ",y_test[i])
    
    
import matplotlib.pyplot as plt

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set Points)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()










