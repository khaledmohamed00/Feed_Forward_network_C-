#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:44:35 2019

@author: khaled
"""
#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import random





def read_data():
    
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)



def read_formated_data():
    
    td,vd,tsd=read_data()
    training_data_input=[np.reshape(x,(784,1)) for x in td[0]]
    training_data_output=[vectorized(y) for y in td[1]]
    training_data=zip(training_data_input,training_data_output)
    validation_inputs = [np.reshape(x, (784, 1)) for x in vd[0]]
    validation_data = zip(validation_inputs, vd[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in tsd[0]]
    test_data = zip(test_inputs, tsd[1])
    return (training_data, validation_data, test_data)
   
    
    
def vectorized(y):

  v=np.zeros((10,1))
  v[y]=1.0
  return v


class ANN(object):
    
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(b,1) for b in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
       
    
    def feedforward(self,a):
        
        for b,w in zip(self.biases,self.weights):
            
            a=sigmoid(np.dot(w,a)+b)
             
        return a;    
            
    def evaluate(self,test_data):
       test_results=[(np.argmax(self.feedforward(x)),y) for x,y in test_data]
       return sum(int(x == y) for (x, y) in test_results)


    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:
                n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)    
            
    
    def update_mini_batch(self,mini_batch,eta):
       
       sum_delta_w=[np.zeros(w.shape) for w in self.weights]
       sum_delta_b=[np.zeros(b.shape) for b in self.biases]
       
       for x,y in mini_batch:
           delta_b,delta_w=self.backprop(x,y)
           sum_delta_w=[ sw+dw for sw,dw in zip(sum_delta_w,delta_w)]
           sum_delta_b=[ sb+db for sb,db in zip(sum_delta_b,delta_b)]
       
       self.weights=[w-(eta/len(mini_batch))*sdw for w,sdw in zip(self.weights,sum_delta_w)]
       self.biases=[b-(eta/len(mini_batch))*sdb for b,sdb in zip(self.biases,sum_delta_b)]
    
    
    def backprop(self,x,y):
       delta_b = [np.zeros(b.shape) for b in self.biases]
       delta_w = [np.zeros(w.shape) for w in self.weights]
       activations=[]
       zs=[]
       z=x
       activation=x
       activations.append(z)
       for w,b in zip(self.weights,self.biases):
           z=np.dot(w,activation)+b 
           zs.append(z)
           activation=sigmoid(z) 
           activations.append(activation) 
       
       delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
       delta_b[-1]=delta
       delta_w[-1]=np.dot(delta,activations[-2].transpose())
       
       for l in range(2,self.num_layers):
           delta=np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(zs[-l])
           delta_b[-l]=delta
           delta_w[-l]=np.dot(delta,activations[-l-1].transpose())        
       
       return (delta_b,delta_w) 
       


   
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
            
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
            
        
training_data, validation_data, test_data=read_formated_data()
#n=ANN([784, 30, 10])
#print n.evaluate(test_data)
#delta_b,delta_w=n.backprop(training_data[0][0],training_data[0][1])
#training_data, validation_data, test_data =load_data_wrapper()

#net = ANN([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)    