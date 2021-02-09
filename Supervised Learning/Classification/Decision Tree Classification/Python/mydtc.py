# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:18:00 2020

@author: vatsal
"""


"""Importing Libraries"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('socialnetworkads.csv')

print("\n")

print(df.shape)

print("\n")

print(df.columns)

print("\n")

print(df.head())

print("\n")

print(df.isnull().sum())

corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr[(corr>=0.5)|(corr<=-0.4)],annot=True,cmap='Blues')


"""Segregating data into X and y"""

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

"""Splitting the dataset"""

from sklearn.model_selection import train_test_split
(X_train,X_test,y_train,y_test)=train_test_split(X,y,test_size=0.3,random_state=0)

print("\n")

print("X Train set: ",X_train)

print("\n")

print("X Test set:",X_test)

print("\n")

print("Y train set:",y_train)

print("\n")

print("Y test:",y_test)


"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print("\n")

print("X Train set: ",X_train)

print("\n")

print("X Test set:",X_test)

print("\n")

"""Calling the model"""
from sklearn.tree import DecisionTreeClassifier
cf=DecisionTreeClassifier(criterion='entropy',random_state=0)
cf.fit(X_train,y_train)

y_pred=cf.predict(X_test)

print("\n")

print("Input Values                       True Result             Predicted Results")

for i in range(0,119):
    print(X_test[i],"               ",y_test[i],"               ",y_pred[i])


from sklearn.metrics import accuracy_score
print("Accuracy of model is",accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report
print("\n")
print(classification_report(y_test,y_pred))








