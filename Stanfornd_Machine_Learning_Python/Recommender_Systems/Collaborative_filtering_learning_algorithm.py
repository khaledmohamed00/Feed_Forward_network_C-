#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 07:26:37 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat

def cost(params,Y,R,no_features,lamda):
    R=np.array(R)
    Y=np.array(Y)
    no_movies=R.shape[0]
    no_users=R.shape[1]
    X=params[:no_movies*no_features].reshape([no_movies,no_features])
    theta=params[no_movies*no_features:].reshape([no_users,no_features])
    
    j=(np.sum(np.power(R*(X.dot(theta.T)-Y),2))+lamda*np.sum(X**2)+lamda*np.sum(theta**2))/2
    
    
    grad_x=(R*(X.dot(theta.T)-Y)).dot(theta)+lamda*X
    grad_theta=(R*(X.dot(theta.T)-Y)).T.dot(X)+lamda*theta
    grad=np.append(grad_x.ravel(),grad_theta.ravel())
    return j,grad
    
def gradient_descent(params,Y,R,no_features,lamda,iterations):
    j_history=[]
    for i in range(iterations):
       
       j,grad=cost(params,Y,R,no_features,lamda)    
       j_history.append(j)
       params=params-lamda*grad
    return params,j_history

def predict(X,theta,y_mean):
    predictions=X.dot(theta.T)
    print predictions.shape
    '''for i in range(X.shape[0]):
      predictions[i,:]=predict[i,:]+y_mean[i]
    '''  
    return predictions  

def normalize(Y,R):
   no_movies=R.shape[0]
   y_mean=np.zeros(no_movies)
   Y_norm=np.zeros(Y.shape)
   for i in range(no_movies):
     idx = np.where(R[i,:] == 1)[0]
     y_mean[i]=Y[idx,:].mean()
     Y_norm[i,:]=Y[i,:]-y_mean[i]
     
     return Y_norm,y_mean
     
data0=loadmat('/mnt/407242D87242D1F8/study/anaconda/Recommender_Systems/data/ex8_movies.mat')
data1=loadmat('/mnt/407242D87242D1F8/study/anaconda/Recommender_Systems/data/ex8_movieParams.mat')

R=np.array(data0['R'])
Y=np.array(data0['Y'])
X=np.array(data1['X'])
theta=np.array(data1['Theta'])
no_movies=R.shape[0]
no_users=R.shape[1]
no_features=100
lamda=0.00001
iterations=400

f = open('/mnt/407242D87242D1F8/study/anaconda/Recommender_Systems/data/movie_ids.txt')

from scipy.optimize import minimize
X = np.random.randn(no_movies, no_features)
Theta = np.random.randn(no_users, no_features)
params = np.append(np.ravel(X), np.ravel(Theta))
Y_norm,y_mean=normalize(Y,R)

fmin = minimize(fun=cost, x0=params, args=(Y_norm, R, no_features,lamda), 
                method='CG', jac=True, options={'maxiter': 200})
params=fmin.x
X=params[:no_movies*no_features].reshape([no_movies,no_features])
theta=params[no_movies*no_features:].reshape([no_users,no_features])     
predictions=predict(X,theta,y_mean)
'''
#params,j_history=gradient_descent(params,Y_norm,R,no_features,lamda,iterations)
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
'''