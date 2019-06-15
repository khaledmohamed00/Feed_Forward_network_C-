#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:16:02 2019

@author: khaled
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2


# use seaborn plotting defaults
import seaborn as sns; sns.set()
from scipy.io import loadmat

def find_closest_centroids(X,centroids):
    c=np.zeros([X.shape[0],X.shape[1]])
    #max_distance=0
    #for i in range(X.shape[2]):

    #  max_distance=max_distance+X[:,:,i].max()**2
    
    for row  in range(X.shape[0]):
        for col  in range(X.shape[1]):
          mini_distance=9999999
          index=0
          mini_index=0
          for center in centroids:
             index=index+1
             distance=0
             for i in range(X.shape[2]):
               distance=distance+((X[row][col][i]-center[i]))**2 
             if distance-mini_distance <0 :
                 mini_distance=distance
                 mini_index=index               
          c[row][col]=mini_index-1
            #c.append(mini_index-1)
    
    return c.astype(int)
         
def compute_centroids(X,c,no_centroids):
    centroids=[]
    for k in range(no_centroids):
        summ=0
        x_sum=np.zeros(X.shape[2])
        for row  in range(X.shape[0]):
          for col  in range(X.shape[1]):
             if k==c[row][col]:
                 summ=summ+1
                 x_sum=x_sum+X[row][col][:]
        centroids.append((1.0*x_sum)/summ)       
    
    return np.array(centroids)


def K_means(X,no_centroids,no_iterations):
    centroids=intialize_centroids(X,no_centroids)
    for i in range(no_iterations):
      c = find_closest_centroids(X, centroids)
      centroids=compute_centroids(X,c,no_centroids)    
    return c,centroids

def intialize_centroids(X,no_centroids):
    centroids=[]   
    randx=np.random.randint(0,high=X.shape[0]-1,size=no_centroids)
    randy=np.random.randint(0,high=X.shape[0]-1,size=no_centroids)
    for x,y in zip(randx,randy):
     centroids.append(X[x][y][:])
    
    return np.array(centroids)

def construct_image(image,c,centroids):
    for row  in range(image.shape[0]):
          for col  in range(image.shape[1]):
              image[row][col][0]=centroids[c[row][col]][0]
              image[row][col][1]=centroids[c[row][col]][1]
              image[row][col][2]=centroids[c[row][col]][2]
    
    return image           



'''
data=loadmat('/mnt/407242D87242D1F8/study/anaconda/K_means_clustering/data/ex7data2.mat')

X=np.array(data['X']).reshape([300,2])

fig,ax=plt.subplots(figsize=(12,8))
#ax.scatter(X[:,0].reshape([300,1]), X[:,1].reshape([300,1]), c='b', s=50, cmap='autumn')
no_centroids=3
no_iterations=20

c,centroids=K_means(X,no_centroids,no_iterations)
cluster1=X[np.where(c==0)]
cluster2=X[np.where(c==1)]
cluster3=X[np.where(c==2)]

ax.scatter(centroids[:,0].reshape([3,1]), centroids[:,1].reshape([3,1]), c='r', s=50, cmap='autumn')
ax.scatter(cluster1[:,0], cluster1[:,1], c='b', s=50, cmap='autumn')
ax.scatter(cluster2[:,0], cluster2[:,1], c='g', s=50, cmap='autumn')
ax.scatter(cluster3[:,0], cluster3[:,1], c='y', s=50, cmap='autumn')
'''
no_centroids=16
no_iterations=10
image_data = loadmat('/mnt/407242D87242D1F8/study/anaconda/K_means_clustering/data/bird_small.mat')
image=(1.0*np.array(image_data['A']).reshape([128,128,3]))/255

c,centroids=K_means(image,no_centroids,no_iterations)
img=construct_image(image,c,centroids)
plt.imshow(img)

