#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:55:01 2017

@author: brian Cechmanek

Implementation of a classifier Perceptron from scratch. with extensive/excessive documentation
We'll later use it for exploration of the Iris data set.
"""

import numpy as np # this is to drastically speed up linear computations

class Perceptron(object):
    """ Perceptron Classifier 
        
        Parameters 
        ----------
        eta: (float) the learning rate. range: 0.0-1.0
        n_iter: (int) number of passes over the training dataset
   
        Attributes
        ----------
        w_ : (1d-array numpy) weights after fitting.
        errors: (list) total number of misclassifications in each epoch.
        
    """
    
    def __init__(self, eta=0.01, n_iters=10): # arbitrarily small starting parameters
        """ instantiate the Perceptrn object with arbitrarily small starting
            parameters. 0.01 is a 'slow' learning rate, and 10 iterations should
            be very quick on small to medium-sized data sets. These parameters
            often require manual optimization.
        """
        self.eta = eta
        self.n_iters = n_iters
        
    def fit(self, X, y):
        """ Fit the training data to the Perceptron. 
        
        Parameters
        ----------
        X: (array-like) defined by n_samples X n_features. 
        y: (array-like) the corresponding target value of each X sample.
        ex: 
            X = [[1,0,1], [0,1,0]]
            y = [ 'Yes' ,  'No'  ]

        Returns
        -------
        self: object
        
        """
        # create an array of zeros the length of 1 + n_features of X
        self.w_ = np.zeros(1 + X.shape[1])
        # and a basic python array to count the misclassifications of each pass
        self.errors_ = []
        
        # actually weight values, by iterating over the set.
        # thus, this is really the time-comsuming step
        for _ in range(self._iters):
            errors = 0
            
            for xi, target in zip(X,y): # xi represents the ith-iteration of a sample in X
                # update the model weight based on diference of prediction to value
                # multiplied by our learning weight (default 0.01)
                # hence why it requires manual tuning to balance speed of 
                # learning and accuracy
                update = self.eta * (target- self.predict(xi)) # see prediction below
                self.w_[1:] += update * xi # update each of the feature weights
                self.w_[0] += update # update the internal (implicit) weight
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self
            
    def net_input(self, X):
        """ Calculate the weighted sum value via dot product of implicit weight, 
            and all feature weights."""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        """ Binary step-function output - if the dot-product of all weights is 
            greater then zero, then return 1. else -1. """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        