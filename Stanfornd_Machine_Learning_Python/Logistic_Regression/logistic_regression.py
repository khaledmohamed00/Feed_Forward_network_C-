#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 03:08:45 2019

@author: khaled
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(x,theta):
    sigmoid(x.dot(theta))
    


def Cost_function(theta,x,y):
      m=len(x)
      x=np.array(x)
      y=np.array(y)
      theta=theta.reshape([1,len(theta)])
      terms=-y*np.log( sigmoid(x.dot(theta.T)))-(1-y)*np.log(1- sigmoid(x.dot(theta.T)))
      
      return np.sum(terms)/m    


def gradient(theta,x,y):
    x=np.array(x)
    y=np.array(y)
    theta=theta.reshape([1,len(theta)])
    #print theta.shape
    grad=np.zeros(theta.shape[1])
    m=len(x)
    error=sigmoid(x.dot(theta.T))-y
    #print sum(error)
    for j in range(theta.shape[1]):
        terms=error*(x[:,j].reshape([m,1]))
        #print error.shape,x[:,j].shape,terms.shape,j
        grad[j]=np.sum(terms)/m
    
    return grad


def predict(x,theta):
   
    theta=theta.reshape([1,len(theta)])

    predictions=sigmoid(x.dot(theta.T))
    result=[1 if i >= 0.5 else 0  for i in predictions ]
    return result
    


path = '/mnt/407242D87242D1F8/study/anaconda/Logistic_Regression/data/ex2data1.txt'
data=pd.read_csv(path,header=None,names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()
data.insert(0, 'Ones', 1)
data.insert(3, 'Exam 1sq', np.power(data['Exam 1'],2))
data.insert(4, 'Exam 2sq', np.power(data['Exam 2'],2))

cols=data.shape[1]
x=np.array(data.iloc[:,0:cols-1])
y=np.array(data.iloc[:,cols-1:cols])

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]



theta=np.zeros(5)

Cost_function(theta,x,y)
import scipy.optimize as opt
result = opt.fmin_tnc(func=Cost_function, x0=theta, fprime=gradient, args=(x, y))
Cost_function(result[0],x,y)
theta=result[0]

x1 = np.linspace(0,100, 100)
#x1 = np.linspace(x[1].min(), x[1].max(), 100)

f = -1*(theta[0] + (theta[1] * x1))/theta[2]
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x1, f, 'r', label='Prediction')
#fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


correct=[1 if (a==b) else 0 for a,b in zip(y,predict(x,theta))]
'''print Cost_function(theta,x,y)
print gradient(theta,x,y)

theta=np.zeros([1,3])
print cost(theta,x, y)
print _gradient(theta,x,y)
'''

