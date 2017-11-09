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
<<<<<<< HEAD
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
            print(len(json_desc['sift']))
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
=======
#Compare 2 different PCA methods
>>>>>>> master
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
    
<<<<<<< HEAD
    return X_kpca
#%%
'''
#plt.plot(Y_x,Y_y, 'ro')

for k in range (0,156):
    if k in range(0,57):
        plt.plot(Y_x[k],Y_y[k], 'ro')
    elif k in range (57,137):
        plt.plot(Y_x[k],Y_y[k], 'bs')
    elif k in range (137,156):
        plt.plot(Y_x[k],Y_y[k], 'g^')

=======
    eig_vals_total = sum(eig_vals)
    var_exp = [(i / eig_vals_total)*100 for i in sorted(eig_vals,reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    cum_var_exp = cum_var_exp[(n_components-1)]
>>>>>>> master
    
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


