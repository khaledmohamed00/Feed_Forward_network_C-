#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 02:19:37 2019

@author: khaled
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat
from sklearn.svm import SVC # "Support vector classifier"

data=loadmat('/mnt/407242D87242D1F8/study/anaconda/support_vector_machine/data/ex6data3.mat')
X=np.array(data['X']).reshape([211,2])
y=np.array(data['y']).reshape([211,1])
Xval=np.array(data['Xval']).reshape([200,2])
yval=np.array(data['yval']).reshape([200,1])

from sklearn.model_selection import GridSearchCV

# Create my estimator and prepare the parameter grid dictionary
params_dict = {"C": np.linspace(1, 100,num=10), "gamma": np.linspace(0.1, 1,num=10)}
svm = SVC(kernel="rbf")

# Fit the grid search
search = GridSearchCV(estimator=svm, param_grid=params_dict,iid=True)
search.fit(X, y.ravel())
print"Best parameter values:", search.best_params_
print"CV Score with best parameter values:", search.best_score_

fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(X[:,0].reshape([211,1]), X[:,1].reshape([211,1]), c=y, s=50, cmap='autumn')

x1 = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1], 100)
x2 = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], 100)
X1, X2 = np.meshgrid(x2, x1)
xx = np.vstack([X1.ravel(), X2.ravel()]).T
model = SVC(kernel='rbf',gamma=search.best_params_['gamma'],C=search.best_params_['C'])
#model = SVC(kernel='rbf',gamma=10,C=10000)

model.fit(X, y.ravel())
#model.score(Xval, yval.ravel())

P = model.decision_function(xx).reshape(X1.shape)
#print P
# plot decision boundary and margins
ax.contour(X1, X2, P, colors='k',levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
print model.score(X,y)
print model.score(Xval, yval.ravel())