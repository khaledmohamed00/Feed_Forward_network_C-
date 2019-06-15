#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 04:10:11 2019

@author: khaled
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    si=1 / (1 + np.exp(-z))
    if si==0:
        return 0.0000001
    
    else:
        return si
    #return 

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def CostReg_function(theta,x,y,beta):
      m=len(x)
      x=np.array(x)
      y=np.array(y)
      
      theta=theta.reshape([1,len(theta)])
      #print x.shape
      #print y.shape
      #print theta.shape
      terms=-y*np.log( sigmoid(x.dot(theta.T)))-(1-y)*np.log(1- sigmoid(x.dot(theta.T)))
      reg=((1.0*beta)/2*m)*np.sum(np.power(theta[:,1:theta.shape[1]],2))

      return np.sum(terms)/m+reg    

def gradientReg(theta,x,y,beta):
    x=np.array(x)
    y=np.array(y)

    theta=theta.reshape([1,len(theta)])
    #print theta.shape
    grad=np.zeros(theta.shape[1])
    m=len(x)
    error=sigmoid(x.dot(theta.T))-y
    #print sum(error)
    grad=(x.T.dot(error)+(beta/2)*theta.T)/(1.0*m)
    
    return grad.flatten()

def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()

def gradient_descent(x,y,theta,alpha,iteration):
    theta=theta.reshape([1,len(theta)])

    t=np.zeros(theta.shape)
    m=len(x)
    cost=[]
    for i in range(iteration):
        
        for j in range(theta.shape[1]):
          t[0,j]=theta[0,j]-alpha*sum((sigmoid(x.dot(theta.T))-y)*(x[:,j]).reshape(m,1))/m
         
        
        for j in range(theta.shape[1]):
         theta[0,j]=t[0,j]
        
        cost.append(CostReg_function(theta,x,y,1))
    return theta,cost    


def g(z) :  # sigmoid function
    return 1.0/(1.0 + np.exp(-z))

def h_logistic(X,theta) : # Model function
    
    return g(np.dot(X,theta.T))

def J(theta,X,y) : # Cost Function 
    #theta=theta.reshape([1,401])

    m = len(y)
    X=np.array(X).reshape([5000,401])
    y=np.array(y).reshape([5000,1])
    theta=theta.reshape([1,X.shape[1]])

    return -(np.sum(np.log(h_logistic(X,theta))) + np.dot((y-1).T,(np.dot(X,theta.T))))/m

def J_reg(theta,X,y,reg_lambda) : # Cost Function with Regularization
    #theta=theta.reshape([1,401])
    theta=theta.reshape([1,len(theta)])

    m = len(y)
    return J(theta,X,y) + reg_lambda/(2.0*m) * np.dot(theta[0][1:],theta[0][1:])

def gradJ(theta,X,y) : # Gradient of Cost Function
    #theta=theta.reshape([1,401])
    theta=theta.reshape([1,X.shape[1]])

    m = len(y)
    return (np.dot(X.T,(h_logistic(X,theta) - y)))/m

def gradJ_reg(theta,X,y,reg_lambda) : # Gradient of Cost Function with Regularization
    #theta=theta.reshape([1,401])
    theta=theta.reshape([1,X.shape[1]])

    m = len(y)
    return (gradJ(theta,X,y)+((reg_lambda/2)*theta.T)/(1.0*m)).flatten() #+ reg_lambda/(2.0*m) * np.concatenate(([0], theta[1:])).T

def oneVSall(X,y,beta,no_label):
    
    theta_all=np.zeros(no_label).tolist()
    for label in range(1,no_label+1):
        
        theta=np.random.randn(401)

        y_i=[1 if (item==label) else 0 for item in y]
        y_i=np.array(y_i).reshape([5000,1])
        while True:
         fmin = minimize(fun=J_reg, x0=theta, args=(X, y_i,beta), method='TNC', jac=gradJ_reg)
         if fmin.success==True:
            break
         else:
             #print 'false',label
             theta=np.random.randn(401)

        if label==10:
            theta_all[0]=fmin.x
            print 'label:',0,'sucessfully trained'

        else:
            theta_all[label]=np.array(fmin.x)
            print 'label:',label,'sucessfully trained' 
    return theta_all
def predict(X,theta_all):
  predicitions=[]
  
  for  x in X:
    #print x  
    x=x.reshape([401,1])
    probabilities=[]
    for theta in theta_all:
      #probabilities=[]
      theta=np.array(theta).reshape([1,401])  
      probability=sigmoid(theta.dot(x))
      probabilities.append(probability)
    #print probabilities
    #print ' '
    max_=np.argmax(probabilities)
    #print max_
    if max_==0:
        predicitions.append(10)
    else:
     predicitions.append(max_)
  
  return predicitions  
    
def evaluate(y,predicitions):
    correct=[1 if (a==b) else 0   for a,b in zip(y,predicitions)]
    return (1.0*sum(correct))/len(y)*100.0
    
data=loadmat('/mnt/407242D87242D1F8/study/anaconda/OneVsAll_logistic_Regression/data/ex4data1.mat')
X=data['X']
y=data['y']
X = np.insert(X, 0, values=np.ones(5000), axis=1)
zero=[1 if (item==10) else 0 for item in data['y']]

X=np.array(X).reshape([5000,401])
y=np.array(y).reshape([5000,1])

beta=1
no_label=10
theta_all=oneVSall(X,y,beta,no_label)
predicitions=predict(X,theta_all)
print 'Training evaluation',evaluate(y,predicitions)

