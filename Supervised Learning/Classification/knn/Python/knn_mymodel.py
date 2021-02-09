# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:50:58 2020

@author: vatsal
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('datasett.csv')

print("\n")

print(df.columns)

print("\n")

print(df.isnull().sum())

print("\n")

corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr[(corr>=0.5)|(corr<=-0.4)],annot=True,cmap='Blues')


"""segregating data into X and y"""

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


"""Splitting the Dataset into training and testing"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)


"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print("\n")

print(X_train)

print("\n")

print(X_test)


"""Calling our main model"""

from sklearn.neighbors import KNeighborsClassifier

cf=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
cf.fit(X_train,y_train)


y_pred=cf.predict(X_test)

from sklearn.metrics import accuracy_score


print("Accuracy is ",accuracy_score(y_test, y_pred)*100)


"""Predicting out of Dataset values"""


print(cf.predict(sc.transform([[25,60000]])))

print("\n")

print(cf.predict(sc.transform([[28,160000]])))


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print("\n")
"""Confusion Matrics"""

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)

print(cm)
















