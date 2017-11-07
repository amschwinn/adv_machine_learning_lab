# -*- coding: utf-8 -*-
"""
Implement kernal trick to be used
in adv computer vision lab 1 comparing 
the following algorithms with their
kernel implememented counterparts
PCA, K-Means, Log Regression,
and SVM

By: Austin Schwinn, 
Jeremie Blanchard

October 10, 2017
"""
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import pandas as pd

#get linear kernel
def linear_kernel(X):
    k = X.dot(X.T)
    return k

#get RBF kernel
def rbf_kernel(X, gamma):
    #Calc pairwise squared euclidean dist for every set of observations
    pair_dist = pdist(X, 'sqeuclidean')
    
    #From pairwise distance to len(X) x len(X) matrix
    symm_pair_dist = squareform(pair_dist)
    
    #Set gamma coeficient
    #gamma = 1
    
    #Use gamma and symmetric pairwise matrix to get kernel matrix
    k = exp(-gamma * symm_pair_dist)
    
    #output kernel Matrix
    return k

#get polynomial kernel
def poly_kernel(X,p):
    k=((X.dot(X.T))+1)**p
    return k

#Get laplacian kernel
def laplacian_kernel(X,gamma):
    #Calc pairwise squared manhattan dist for every set of observations
    pair_dist=pdist(X,'cityblock')
    
    #From pairwise distance to len(X) x len(X) matrix
    symm_pair_dist = squareform(pair_dist)
    
    #Set gamma coeficient
    #gamma = 1
    
    #Use gamma and symmetric pairwise matrix to get kernel matrix
    k = exp(-gamma * symm_pair_dist)
    
    #output kernel Matrix
    return k

#Center kernel matrix
def center_kernel(k):
    #Center kernel matrix according to kernel trick from class
    #Kcentered=K−1nK−K1n+1nK1n=(I−1n)K(I−1n)
    n = np.ones((k.shape[0],k.shape[0])) / k.shape[0]
    k_centered = k-n.dot(k)-k.dot(n)+n.dot(k).dot(n)
    
    return k_centered