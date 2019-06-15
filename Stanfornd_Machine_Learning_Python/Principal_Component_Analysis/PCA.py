#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:11:58 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
#import cv2
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat
import cv2
def normalize(X):
    X=(X-X.mean())/X.std()
    return X
    

def PCA(X):
    
    cov=(X.T.dot(X))/X.shape[0]
    U,S,V=np.linalg.svd(cov)
    return U,S,V
def data_projection(U,no_p,X):
    #Z=U[:,0:no_p].dot(X.T)
    Z=X.dot(U[:,0:no_p])
    return Z

def data_recovery(Z,U,no_p):    
    X_recovered=Z.dot(U[:,0:no_p].T)
    
    return X_recovered


'''
data=loadmat('/mnt/407242D87242D1F8/study/anaconda/Principal_Component_Analysis/data/ex7data1.mat')
X=np.array(data['X'])

fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1], c='b', s=50, cmap='autumn')
no_p=1
X_norm=normalize(X)
fig,ax=plt.subplots(figsize=(12,8))
U,S,V=PCA(X)
Z=data_projection(U,no_p,X)
X_recovered=data_recovery(Z,U,no_p)
#ax.scatter(X_norm[:,0], X_norm[:,1], c='b', s=50, cmap='autumn')
ax.scatter(X_recovered[:,0], X_recovered[:,1], c='r', s=50, cmap='autumn')
'''
data=loadmat('/mnt/407242D87242D1F8/study/anaconda/Principal_Component_Analysis/data/ex7faces.mat')
X=data['X']

X_norm=normalize(X)
no_p=100
U,S,V=PCA(X_norm)
Z=data_projection(U,no_p,X_norm)
X_recovered=data_recovery(Z,U,no_p)
img=X_recovered[3,:].reshape([32,32])
plt.imshow(img)

plt.imshow(img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
