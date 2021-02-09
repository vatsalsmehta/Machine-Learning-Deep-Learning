# -*- coding: utf-8 -*-
"""
Spyder Editor

Author:Vatsal Mehta.
"""

import numpy as np
import pandas as pd

df=pd.read_csv('Data.csv')
print("\n")

print(df.columns)

print("\n")

print(df.shape)

print("\n")

print(df.corr())
print("\n")

print("Max Salary value is in row number: ",df['Salary'].idxmax()) #it will print the  row number of maximum value in that column

print("Salary to age ratio is",df['Salary']/df['Age'])

print("\n")
print(df.isnull().sum())

print("\n")
print(df['Salary'].isnull())


#Different Methods to fill missing Data

#method 1 

df.loc[4,'Salary']=40000

print(df)

#method 2

df['Age'].fillna(45,inplace=True)

print(df)

#method 3

df.dropna(thresh=3)


#Segregation of input and output variables

X=df.iloc[:,0:3].values
y=df.iloc[:,3].values


#method 4

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

print("\n")


"""Encoding Categorical Data for input data present in X variable """

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X)

"""One hot Encoder basically splits one single column into multiple columns having numerical values"""

print("\n")

"""Encoding Categorical Data for output data present in y variable """


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(y)

"""Label Encoder basically labels string values into numerical values and does not split the columns"""


"""Splitting dataset into training and test set"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

"""
    X is always used by me for input variables and y for output variables
    train for training and test for testing


    X_train= conatins input data for training from variable X
    X_test= contain input data for testing from variable X
    y_train=contains output data for training from variable y
    y_test=contains output data for testing from variable y """
    
    
print(X_train)

print("\n")

print(X_test)

print("\n")
print(y_train)

print("\n")
print(y_test)

print("\n")

"""Feature Scaling using StandardScaler"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])

print(X_train)

print("\n")

print(X_test)

print("\n")
    




















