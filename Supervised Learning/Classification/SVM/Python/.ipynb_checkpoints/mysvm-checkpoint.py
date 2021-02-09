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
X_test=sc.fit_transform(X_test)


from sklearn.svm import SVC

cf=SVC(kernel='linear',random_state=0)
cf.fit(X_train,y_train)

print("last")

"""Testing model"""

y_pred=cf.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

print("Accuracy of model is",accuracy_score(y_test, y_pred)*100)

print("\n")

cm=confusion_matrix(y_test, y_pred)
print(cm)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, cf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, cf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()









