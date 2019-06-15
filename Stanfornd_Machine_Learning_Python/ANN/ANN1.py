#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:40:17 2019

@author: khaled
"""

import json
import random
import sys
import cPickle
import gzip
from matplotlib import pyplot as plt

# Third-party libraries
import numpy as np

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

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

class ANN1(object):
    def __init__(self,sizes,cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self,a):
        
        for b,w in zip(self.biases,self.weights):
            
            a=sigmoid(np.dot(w,a)+b)
             
        return a;    
            
    def evaluate(self,test_data):
       test_results=[(np.argmax(self.feedforward(x)),y) for x,y in test_data]
       return sum(int(x == y) for (x, y) in test_results)


    def SGD(self,training_data,epochs,mini_batch_size,eta,lmbda = 0.0,test_data=None,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        
        n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        if test_data:
                n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta,lmbda, len(training_data))
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)    
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
    
    def update_mini_batch(self,mini_batch,eta,lmbda, n):
       
       sum_delta_w=[np.zeros(w.shape) for w in self.weights]
       sum_delta_b=[np.zeros(b.shape) for b in self.biases]
       
       for x,y in mini_batch:
           delta_b,delta_w=self.backprop(x,y)
           sum_delta_w=[ sw+dw for sw,dw in zip(sum_delta_w,delta_w)]
           sum_delta_b=[ sb+db for sb,db in zip(sum_delta_b,delta_b)]
       
       self.weights=[(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*sdw for w,sdw in zip(self.weights,sum_delta_w)]
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
       
       delta=(self.cost).delta(zs[-1], activations[-1], y)
       delta_b[-1]=delta
       delta_w[-1]=np.dot(delta,activations[-2].transpose())
       
       for l in range(2,self.num_layers):
           delta=np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_prime(zs[-l])
           delta_b[-l]=delta
           delta_w[-l]=np.dot(delta,activations[-l-1].transpose())        
       
       return (delta_b,delta_w) 
   
    def accuracy(self, data, convert=False):
        
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
       
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()



def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ANN1(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


training_data, validation_data, test_data=read_formated_data()
net = ANN1([784, 30, 10], cost=CrossEntropyCost)
evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy=net.SGD(training_data, 100, 10, 0.5,lmbda = 5.0,test_data=test_data,evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)
net.save("network")
x=range(0,100)
plt.plot(x,training_cost)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()  
plt.plot(x,evaluation_cost)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()   

plt.plot(x,training_accuracy)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()   

plt.plot(x,evaluation_accuracy)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()     