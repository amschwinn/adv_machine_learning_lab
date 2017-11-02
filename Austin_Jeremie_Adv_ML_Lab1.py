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
import numpy as np
import pandas as pd

#Load our built lab functions
import kernels
import PCA
import kmeans

#%%
#Use the IRIS DATASET
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')
#%%
#set column names
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
# drops the empty line at file-end
df.dropna(how="all", inplace=True) 

#Split features and label
X = df.ix[:,0:4].values
Y = df.ix[:,4]

#Standardize data in order to get mean=0 and the variance=1
X_std = StandardScaler().fit_transform(X)

#Get linear kernel
linear_k = kernels.linear_kernel(X_std)

#%%
#Run kernel matrix in kmeans
k_centroid, k_clusters = kmeans.kmeans(linear_k,3)


#%%
np.min(linear_k)
np.max(linear_k)