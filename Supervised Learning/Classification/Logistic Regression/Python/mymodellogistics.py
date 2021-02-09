# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:31:18 2020

@author: vatsal
"""
import numpy as np
import pandas as pd

df=pd.read_csv('Social_Network_Ads.csv')


print("\n")

print(df.columns)

print("\n")

print(df.shape)

print("\n")

print(df.isnull().sum())

print("\n")

print(df.corr())

import seaborn as sns
import matplotlib.pyplot as plt
corr=df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr[(corr>=0.5)|(corr<=-0.4)],annot=True,cmap='Blues')


"""Segregate data into X and y"""
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


"""Splitting the dataset"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


print(X_train)
print("\n")

print(X_test)
print("\n")

print(y_train)
print("\n")

print(y_test)
print("\n")


"""Feature Scaling"""
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


print(X_train)
print("\n")

print(X_test)
print("\n")



from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print(y_pred)

print("\n")
print("Predicted Values    Actual Values ")

for i in range(0,len(y_pred)):
    print(y_pred[i],"                    ",y_test[i])

print("\n")


#Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score

print("Confusion Matrix")
cm=confusion_matrix(y_test, y_pred)
print(cm)
print("\n")

# Predicting a new result
print(clf.predict(sc.transform([[30,87000]])))

"""Accuracy"""
print("The Accuracy of our model is ",accuracy_score(y_test,y_pred)*100)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


















