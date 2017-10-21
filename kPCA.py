# -*- coding: utf-8 -*-
"""
Using kernal trick on 
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
#%%
#Use the IRIS DATASET
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

#set column names
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
# drops the empty line at file-end
df.dropna(how="all", inplace=True) 

#Split features and label
X = df.ix[:,0:4].values
Y = df.ix[:,4]

#Standardize data in order to get mean=0 and the variance=1
X_std = StandardScaler().fit_transform(X)

#Calc pairwise squared euclidean dist for every set of observations
pair_dist = pdist(X_std, 'sqeuclidean')

#From pairwise distance to len(X) x len(X) matrix
symm_pair_dist = squareform(pair_dist)

#Set gamma coeficient
gamma = 1

#Use gamma and symmetric pairwise matrix to get kernel matrix
k = exp(-gamma * symm_pair_dist)

#Center kernel matrix according to kernel trick from class
#Kcentered=K−1nK−K1n+1nK1n=(I−1n)K(I−1n)
n = np.ones((k.shape[0],k.shape[0])) / k.shape[0]
k_centered = k-n.dot(k)-k.dot(n)+n.dot(k).dot(n)

#Use centered k matrix to get eignvectors & eigenvalues from centered k
eigvals, eigvecs = eigh(k_centered)

#Specify number of components we want to test
n_components = 2

#Keep eienvectors according to number of components with highest eigenvalues
X_kpca = np.column_stack((eigvecs[:,-comp] for comp in range(1,n_components+1)))
