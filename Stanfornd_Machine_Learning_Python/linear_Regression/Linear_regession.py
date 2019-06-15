#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 04:32:15 2019

@author: khaled
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def h(x,theta):

    return x.dot(theta.transpose())
    
def compute_cost(x,y,theta):
    m=len(x)
    term=h(x,theta)-y
    return sum(np.power(term,2))/(2*m)

def gradient_descent(x,y,theta,alpha,iteration):
    t=np.zeros(theta.shape)
    m=len(x)
    cost=[]
    for i in range(iteration):
        
        for j in range(theta.shape[1]):
          t[0,j]=theta[0,j]-alpha*sum((h(x,theta)-y)*(x[:,j]).reshape(m,1))/m
         
        
        for j in range(theta.shape[1]):
         theta[0,j]=t[0,j]
        
        cost.append(compute_cost(x,y,theta))
    return theta,cost    
def predict(x,theta,mean,std):
    noramlized_value =h(np.array([1,(x[0]-mean[0])/std[0],(x[1]-std[1])/std[1]]),theta)
    precdited_value=noramlized_value*std[2]+mean[2]
    return precdited_value

data1 = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#data1.insert(0,'ones',1)
cols=data1.shape[1]
x1=np.array(data1.iloc[:,0:cols-1])
y1=np.array(data1.iloc[:,cols-1:cols])
theta1=np.zeros([1,2])
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x1, y1)
f = model.predict(x1)
x = np.array(x1[:, 0])
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data1.Population, data1.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')



'''
data2 = pd.read_csv('ex1data2.txt', header=None, names=['size','bedrooms', 'prices'])
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
mean=data2.mean()
std=data2.std()
data2=(data2-data2.mean())/data2.std()

data2.insert(0,'ones',1)
cols=data2.shape[1]
x2=np.array(data2.iloc[:,0:cols-1])
y2=np.array(data2.iloc[:,cols-1:cols])
theta2=np.zeros([1,3])

    
#print cost(x,y,theta)
theta2,cost2=gradient_descent(x2,y2,theta2,0.01,1500)

'''
'''
l = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + (theta[0, 1] * l)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(l, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size') 
'''

'''
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1500), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')   
h(np.array([1,(2104-mean[0])/std[0],(3-std[1])/std[1]]),theta2)
'''