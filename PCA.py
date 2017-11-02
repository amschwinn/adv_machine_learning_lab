# -*- coding: utf-8 -*-
"""
Implement PCA and kPCA 
to compare in advance machine learning lab 1

By: Austin Schwinn, 
Jeremie Blanchard

October 10, 2017
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import pandas as pd

import pymysql
import pymysql.cursors
import json

import matplotlib.pyplot as plt

#%%
'''
##Use our SIFT descriptors
# Connect to the database.
conn = pymysql.connect(db='images_db', user='root', passwd='', host='localhost')

sql_get_descriptors = "SELECT * FROM `desc_obj`"

with conn.cursor() as cursor:
    cursor.execute(sql_get_descriptors) #We execute our SQL request
    conn.commit()
    
    it = 0
    list_conca=[]
    
    for row in cursor:
        if it < 3:
            json_desc = json.loads(row[2])
            list_conca = list_conca + json_desc['sift']
            print("Passage n°",it)
            it+=1
        else:
           break
       
    #Move list of arrays to 2d df
    df = pd.DataFrame(list_conca)
    # split data table into data X and class labels y
    X = df.ix[:,0:64].values
'''
#%%
'''
#Use the IRIS DATASET
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()
X = df.ix[:,0:4].values
Y = df.ix[:,4]

#Standardize data in order to get mean=0 and the variance=1
X_std = StandardScaler().fit_transform(X)
'''
#%%
#We are getting the covariance matrix for this dataset
def PCA(X_std):
    cov_mat = np.cov(X_std.T)
    print('NumPy covariance matrix:')
    print(cov_mat)

    #We are extracting the eigenvalues and eigenvectors from our covariance 
    #matrix in order to find the mlst informations features
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)


    #To find which features contains the most informations we need to find the 
    #biggest eigenvalues
    #Because if the eigenvalues is low it means that the feature contain a lot 
    #of information about the dataset
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in 
                 range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    #eig_pairs.sort()
    #eig_pairs.reverse()

    # Visually confirm that the list is correctly sorted by decreasing 
    #eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    print(tot)
    print(cum_var_exp)


    matrix_w = np.hstack((eig_pairs[0][1].reshape(64,1), 
                          eig_pairs[1][1].reshape(64,1)))
    
    print('Matrix W:\n', matrix_w)
    
    Y_ = X_std.dot(matrix_w)
    
    Y_x = Y_[:,0]
    Y_y =Y_[:,1]
    
    print(min(Y_x))
    print(max(Y_x))
    
    print(min(Y_y))
    print(max(Y_y))

    return matrix_w

#Get kPCA
def kpca(k_centered, n_components):
    #Use centered k matrix to get eignvectors & eigenvalues from centered k
    eigvals, eigvecs = eigh(k_centered)
    
    #Specify number of components we want to test
    #n_components = 2
    
    #Keep eienvectors according to number of components with highest eigenvalues
    X_kpca = np.column_stack((eigvecs[:,-comp] for comp in range(1,n_components+1)))
    
    return X_kpca
#%%
'''
#plt.plot(Y_x,Y_y, 'ro')

for k in range (0,1):
    if k in range(0,1):
        plt.plot(Y_x[k],Y_y[k], 'ro')
    elif k in range (1,2):
        plt.plot(Y_x[k],Y_y[k], 'bs')
    elif k in range (2,3):
        plt.plot(Y_x[k],Y_y[k], 'g^')

    
plt.axis([-9, 9, -9, 9])
plt.show()
'''


