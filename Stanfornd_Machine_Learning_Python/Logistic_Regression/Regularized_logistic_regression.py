#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:05:57 2019

@author: khaled
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x,theta):
    sigmoid(x.dot(theta))
    


def CostReg_function(theta,x,y,beta):
      m=len(x)
      x=np.array(x)
      y=np.array(y)
      y=y.reshape(118,1)

      theta=theta.reshape([1,len(theta)])
      terms=-y*np.log( sigmoid(x.dot(theta.T)))-(1-y)*np.log(1- sigmoid(x.dot(theta.T)))
      reg=(beta/2*m)*np.sum(np.power(theta[:,1:theta.shape[1]],2))
      return np.sum(terms)/m+reg    


    
def gradientReg(theta,x,y,beta):
    x=np.array(x)
    y=np.array(y)
    y=y.reshape(118,1)

    theta=theta.reshape([1,len(theta)])
    grad=np.zeros(theta.shape[1])
    m=len(x)
    error=sigmoid(x.dot(theta.T))-y
    for j in range(theta.shape[1]):
        terms=error*(x[:,j].reshape([m,1]))
        if j==0:
         grad[j]=np.sum(terms)/m
        else:
         grad[j]=np.sum(terms)/m+(beta/m)*theta[0][j]
    return grad



def predict(x,theta):
   
    theta=theta.reshape([1,len(theta)])

    predictions=sigmoid(x.dot(theta.T))
    result=[1 if i >= 0.5 else 0  for i in predictions ]
    return result

def evaluate(x,y,theta):
    correct=[1 if (a==b) else 0 for a,b in zip(y,predict(X,theta))]
    accuracy = 100.0*(sum(correct) / (1.0*len(correct)))
    return accuracy



path = '/mnt/407242D87242D1F8/study/anaconda/Logistic_Regression/data/ex2data2.txt'
data=pd.read_csv(path,header=None,names=['test1', 'test2', 'accepted'])

positive=data[data['accepted'].isin([1])]
negative=data[data['accepted'].isin([0])]

x1 = data['test1']
x2 = data['test2']
data.insert(0, 'Ones', 1)
beta=1

cols=data.shape[1]

x=np.array(data.iloc[:,0:cols-1])
X = np.column_stack( (x[:,0], x[:,1],x[:,2],x[:,1]*x[:,2], x[:,1]**2,x[:,1]**2,x[:,1]**3,x[:,1]**3  )) # construct the augmented matrix X
theta=np.zeros(X.shape[1])
y=np.array(data.iloc[:,cols-1:cols])
y=y.reshape(118,1)


import scipy.optimize as opt
result = opt.fmin_tnc(func=CostReg_function, x0=theta, fprime=gradientReg, args=(X, y,beta))
CostReg_function(result[0],X,y,beta)
theta=result[0]

theta=np.array(result[0])
xb1 = np.linspace(-5.0, 5.0, 100)
xb2 = np.linspace(-5.0, 5.0, 100)
Xb1, Xb2 = np.meshgrid(xb1,xb2)

fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(positive['test1'],positive['test2'], s=50, c='b', marker='o', label='accepted')
ax.scatter(negative['test1'],negative['test2'], s=50, c='r', marker='o', label='not accepted')
ax.legend()
ax.set_xlabel('test 1 Score')
ax.set_ylabel('test 2 Score')
#b = a+b*(Xb1) +c*(Xb2)+d*Xb1*Xb2+e*(Xb1**2) +f*(Xb2**2)+g*(Xb1**3) +h*(Xb2**3) 
b = theta[0]+theta[1]*(Xb1) +theta[2]*(Xb2)+theta[3]*Xb1*Xb2+theta[4]*(Xb1**2) +theta[5]*(Xb2**2)+theta[6]*(Xb1**3) +theta[7]*(Xb2**3) 

ax.contour(Xb1,Xb2,b,[0], colors='r')
plt.title("Decision Boundary", fontsize=24)
ax.axis('equal')

