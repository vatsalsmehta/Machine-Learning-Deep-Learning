# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:59:31 2020

@author: vatsal
"""



import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt



df=pd.read_csv('sna.csv')


print("\n")

print(df.columns)


print("\n")

print(df.shape)


print("\n")

print(df.isnull().sum())

"""Heatmap"""
corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr[(corr>=0.5)|(corr<=-0.4)],annot=True,cmap='Blues')


"""Segregate Data into X and y"""

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


"""Feature Scaling"""
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.svm import SVC

cf=SVC(kernel='rbf',random_state=0)
cf.fit(X_train,y_train)

print("\n")

"""Testing model"""

y_pred=cf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

print("Accuracy of model is",accuracy_score(y_test, y_pred)*100)

print("\n")

cm=confusion_matrix(y_test, y_pred)
print(cm)

print("\n")


from sklearn.metrics import classification_report
print("Detailed Analysis of our mode\n\n",classification_report(y_test,y_pred))














