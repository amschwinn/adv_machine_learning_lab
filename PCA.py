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
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

#%%
#Compare 2 different PCA methods
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

#Get PCA
def pca(x, n_components, need_cov = False):
    if need_cov == True:
        x = np.cov(x)
    #Use centered k matrix to get eignvectors & eigenvalues from centered k
    eig_vals, eig_vecs = eigh(x)
    
    #Keep eienvectors according to number of components with highest eigenvalues
    X_kpca = np.column_stack((eig_vecs[:,-comp] for comp in range(1,
                              n_components+1)))
    
    eig_vals_total = sum(eig_vals)
    var_exp = [(i / eig_vals_total)*100 for i in sorted(eig_vals,reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    cum_var_exp = cum_var_exp[(n_components-1)]
    
    return X_kpca, cum_var_exp
#%%
def pca_plots(k_trick,generator,x,y,pca,k_pca):
    #Plot original dataset
    #plot swiss roll in 3d
    if generator.__name__ == 'make_swiss_roll':
        chart = plt.figure()
        ax = chart.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.rainbow)
        name = str(generator.__name__) + ' artificial dataset'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()
    #other benchmarks in 2d
    else:
        plt.figure()
        plt.scatter(x[y==1, 0], x[y==1, 1], alpha=.75, color='red')
        plt.scatter(x[y==0, 0], x[y==0, 1], alpha=.75, color='blue')
        name = str(generator.__name__) + ' artificial dataset'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()

    #Plot non-kernilized PCA
    #plot swiss roll in 3d
    if generator.__name__ == 'make_swiss_roll':
        #1st PC
        plt.figure()
        plt.scatter(pca[:, 0], np.zeros((len(pca[:, 0]),1)), alpha=.75,
                    cmap=plt.cm.rainbow, c=y)
        name = str(generator.__name__) + ' non-kernilized first PC'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()

        #1st and 2nd PCs
        plt.figure()
        plt.scatter(pca[:, 0], pca[:, 1], alpha=.75, c=y, 
                    cmap=plt.cm.rainbow)
        name = str(generator.__name__) + ' non-kernilized first 2 PCs'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.xlabel('First PC')
        plt.ylabel('Second PC')
        plt.savefig(filename)
        plt.close()
            
    #All other benchmark dataset    
    else: 
        #1st PC
        plt.figure()
        plt.scatter(pca[y==1, 0], np.zeros((len(pca[y==1, 0]),1)), 
                    alpha=.75, color='red')
        plt.scatter(pca[y==0, 0], np.zeros((len(pca[y==0, 0]),1)), 
                    alpha=.75, color='blue')
        name = str(generator.__name__) + ' non-kernilized first PC'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()

        #1st and 2nd PCs
        plt.figure()
        plt.scatter(pca[y==1, 0], pca[y==1, 1], alpha=.75, color='red')
        plt.scatter(pca[y==0, 0], pca[y==0, 1], alpha=.75, color='blue')
        name = str(generator.__name__) + ' non-kernilized first 2 PCs'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.xlabel('First PC')
        plt.ylabel('Second PC')
        plt.savefig(filename)
        plt.close()

    #Plot kernilized k_pca
    #plot swiss roll in 3d
    if generator.__name__ == 'make_swiss_roll':
        #1st PC
        plt.figure()
        plt.scatter(k_pca[:, 0], np.zeros((len(k_pca[:, 0]),1)), alpha=.75,
                    cmap=plt.cm.rainbow, c=y)
        name = str(generator.__name__) + ' ' + str(k_trick.__name__) + \
                  ' first PC'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()

        #1st and 2nd PCs
        plt.figure()
        plt.scatter(k_pca[:, 0], k_pca[:, 1], alpha=.75, c=y, 
                    cmap=plt.cm.rainbow)
        name = str(generator.__name__) + ' ' + str(k_trick.__name__) + \
                  ' first 2 PCs'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.xlabel('First PC')
        plt.ylabel('Second PC')
        plt.savefig(filename)
        plt.close()     
        
    else:
        #1st PC
        plt.figure()
        plt.scatter(k_pca[y==1, 0], np.zeros((len(k_pca[y==1, 0]),1)), 
                    alpha=.75, color='red')
        plt.scatter(k_pca[y==0, 0], np.zeros((len(k_pca[y==0, 0]),1)), 
                    alpha=.75, color='blue')
        name = str(generator.__name__) + ' ' + str(k_trick.__name__) + \
                  ' first PC'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.savefig(filename)
        plt.close()

        #1st and 2nd PCs
        plt.figure()
        plt.scatter(k_pca[y==1, 0], k_pca[y==1, 1], alpha=.75, color='red')
        plt.scatter(k_pca[y==0, 0], k_pca[y==0, 1], alpha=.75, color='blue')
        name = str(generator.__name__) + ' ' + str(k_trick.__name__) + \
                  ' first 2 PCs'
        filename = 'outputs/pca/' + name
        plt.title(name)
        plt.xlabel('First PC')
        plt.ylabel('Second PC')
        plt.savefig(filename) 
        plt.close()


