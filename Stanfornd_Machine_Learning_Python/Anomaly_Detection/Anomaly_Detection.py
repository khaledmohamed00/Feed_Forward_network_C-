#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:20:14 2019

@author: khaled
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat



def estimate_gaussian(X):
    mu=[X[:,0].mean(),X[:,1].mean()]
    sgma=[X[:,0].var(),X[:,1].var()]
    
    return np.array(mu),np.array(sgma)

def select_threshold(Xval,yval,mu,sgma):
    #P=univariateGaussian(X,mu,sigma)
    P= multivariateGaussian_victorized(Xval, mu, sgma)
    
    step = (P.max() - P.min()) / 1000
    max_F1=0
    max_threshold=0
    for threshold in np.arange(P.min(),P.max(),step) :        
        ypredict=[1 if(p<threshold) else 0 for p in P]
        tp=[ 1 if(a==b==1) else 0   for a,b in zip(ypredict,yval) ]
        fp=[1 if(a!=b==0) else 0   for a,b in zip(ypredict,yval) ]
        fn=[1 if(a!=b==1) else 0   for a,b in zip(ypredict,yval) ]
         
        
        
        try:   
          prec=1.0*sum(tp)/(sum(tp)+sum(fp))
          rec=1.0*sum(tp)/(sum(tp)+sum(fn))
          F1=1.0*(2*prec*rec)/(prec+rec)
        except (TypeError, ZeroDivisionError):
            continue 
        if F1>max_F1:
           max_F1=F1
           max_threshold=threshold
           
          
    print max_F1
    print max_threshold
    return max_F1,max_threshold
    


def multivariateGaussian_victorized(X, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.
    """
    k = len(mu)
    
    
    sigma2=np.diag(sigma2)
    X = X - mu
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5))*np.exp(-0.5*np.sum(np.multiply(X.T,np.linalg.pinv(sigma2).dot(X.T)),axis=0))
    return p


def multivariateGaussian(X, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.
    """
    k = len(mu)
    
    
    sigma2=np.diag(sigma2)
    X = X - mu
    X=X.reshape([2,1])
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5))*np.exp(-0.5*np.dot(X.T,np.linalg.pinv(sigma2)).dot(X))
    return p
def univariateGaussian(X,mu,sigma):
    
    p0=stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
    p1=stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])
    P=p0*p1
    return P

def predict(Probabilities,threshold):
    ypredict=[1 if(p<threshold) else 0 for p in Probabilities]

    return np.array(ypredict)

data=loadmat('/mnt/407242D87242D1F8/study/anaconda/Anomaly_Detection/data/ex8data1.mat')
X=np.array(data['X'])
Xval=np.array(data['Xval'])
yval=np.array(data['yval'])

mu,sigma=estimate_gaussian(X)
#P=multivariateGaussian(Xval,mu, sigma)

F1,threshold=select_threshold(Xval,yval,mu,sigma)
P=multivariateGaussian_victorized(Xval,mu, sigma)
ypredict=predict(P,threshold)
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(Xval[np.where(ypredict == 1),0], Xval[np.where(ypredict == 1),1], c='r', s=50, cmap='autumn')
ax.scatter(Xval[np.where(ypredict == 0),0], Xval[np.where(ypredict == 0),1], c='b', s=50, cmap='autumn')

x1 = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1], 100)
x2 = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1], 100)
X1, X2 = np.meshgrid(x2, x1)
xx = np.vstack([X1.ravel(), X2.ravel()]).T
P=multivariateGaussian_victorized(xx,mu, sigma)
P=P.reshape(X1.shape)
ax.contour(X1, X2, P, colors='k', alpha=0.5)

