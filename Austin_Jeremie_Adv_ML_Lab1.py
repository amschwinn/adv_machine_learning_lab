# -*- coding: utf-8 -*-
"""
Using kernal trick with
PCA, K-Means, Log Regression,
and SVM on IRIS datasets

By: Austin Schwinn & 
Jeremie Blanchard

October 10, 2017
"""
#Load premade modules
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

#Load our built lab functions
import kernels as AJ_kernels
import PCA as AJ_PCA
import kmeans as AJ_kmeans

#%%
###############################################################################
#Load and prepare data
###############################################################################
#Load the IRIS DATASET
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(
    filepath_or_buffer=url, 
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

#%%
###############################################################################
#Use kernel tricks to move iris dataset into hyperdimensional plane
###############################################################################

#Linear kernel
linear_k = AJ_kernels.linear_kernel(X_std)

#RBF kernel
rbf_k = AJ_kernels.rbf_kernel(X_std,1)

#Polynomial kernel
poly_k = AJ_kernels.poly_kernel(X_std,2)

#laplacian kernel
laplacian_k = AJ_kernels.laplacian_kernel(X_std,1)

#Center the kernels
linear_k_center = AJ_kernels.center_kernel(linear_k)
rbf_k_center = AJ_kernels.center_kernel(rbf_k)
poly_k_center = AJ_kernels.center_kernel(poly_k)
laplacian_k_center = AJ_kernels.center_kernel(laplacian_k)

#%%
###############################################################################
#Lab section 1
#Compare PCA with kernelized PCA
###############################################################################

#Normal PCA
pca = AJ_PCA.PCA(X_std)

#Linear kernel PCA
linear_pca = AJ_PCA.PCA(linear_k_center)

#rbf kernel PCA
rbf_pca = AJ_PCA.PCA(rbf_k_center)

#poly kernel PCA
poly_pca = AJ_PCA.PCA(poly_k_center)

#laplacian kernel PCA
laplacian_pca = AJ_PCA.PCA(laplacian_k_center)

#%%
###############################################################################
#Lab section 2
#Compare K-Means with kernelized K-Means
###############################################################################

#Normal Kmeans
centroids,clusters = AJ_kmeans.kmeans(X_std,3)

#Linear kernel Kmeans
linear_centroids,linear_clusters = AJ_kmeans.kmeans(linear_k_center,3)

#rbf kernel Kmeans
rbf_centroids,rbf_clusters = AJ_kmeans.kmeans(rbf_k_center,3)

#poly kernel Kmeans
poly_centroids,poly_clusters = AJ_kmeans.kmeans(poly_k_center,3)

#laplacian kernel Kmeans
laplacian_centroids,laplacian_clusters = AJ_kmeans.kmeans(laplacian_k_center,3)


#%%
###############################################################################
#Lab section 3
#Compare Logistic Regression with kernelized Logistic Regression
###############################################################################

#Normal logistic regression
reg_score = cross_val_score(LogisticRegression(), X_std, Y, 
                            scoring='accuracy',cv=5)

#linear kernel logistic regression
linear_reg_score = cross_val_score(LogisticRegression(), linear_k_center, Y, 
                            scoring='accuracy',cv=5)

#rbf kernel logistic regression
rbf_reg_score = cross_val_score(LogisticRegression(), rbf_k_center, Y, 
                            scoring='accuracy',cv=5)

#poly kernel logistic regression
poly_reg_score = cross_val_score(LogisticRegression(), poly_k_center, Y, 
                            scoring='accuracy',cv=5)

#laplacian kernel logistic regression
laplacian_reg_score = cross_val_score(LogisticRegression(), laplacian_k_center, 
                            Y, scoring='accuracy',cv=5)
