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
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.cluster import KMeans


#Load our built lab functions
import kernels as AJ_kernels
import PCA as AJ_PCA
import kmeans as AJ_kmeans


#%%
###############################################################################
#Lab section 1
#Compare PCA with kernelized PCA
###############################################################################

#Creat list of Kernel tricks to iterate through
kernels = [AJ_kernels.linear_kernel,AJ_kernels.rbf_kernel,
           AJ_kernels.poly_kernel,AJ_kernels.laplacian_kernel]

benchmark_kernels = ["linear","rbf","poly"]

data_benchmarks = [make_moons,make_circles,make_swiss_roll,make_classification]

run_times = {}
variance_explained = {}

#Iterate through kernels to compare with non-kernelized PCA
for k_trick in kernels:
    #Iterate through data benchmarks
    for generator in data_benchmarks:
        #Generate data
        x, y = generator(n_samples=250, random_state=613)

        #start run timer
        start = timeit.default_timer()

        #start by kernelizing data
        #Add params when necessary
        if (k_trick.__name__ == "linear_kernel"):
            #start run timer
            start = timeit.default_timer()
            k = k_trick(x)
        elif (k_trick.__name__ == "poly_kernel"):
            #find best p value
            var_cum = []
            poly_var = []
            for poly in range(1,20):
                k = k_trick(x,poly)
                k_center = AJ_kernels.center_kernel(k)
                k_pca, k_var_explain = AJ_PCA.pca(k_center,n_components=2)
                var_cum = var_cum + [k_var_explain]
                poly_var = poly_var + [poly]
            #Use p value that had highest variance expalined
            #poly = poly_var[var_cum.index(max(var_cum))]
            #start run timer
            start = timeit.default_timer()
            poly = 7
            k = k_trick(x,poly)
        else:
            var_cum = []
            gamma_var = []
            for gamma in range(1,50):
                k = k_trick(x,gamma)
                k_center = AJ_kernels.center_kernel(k)
                k_pca, k_var_explain = AJ_PCA.pca(k_center,n_components=2)
                var_cum = var_cum + [k_var_explain]
                gamma_var = gamma_var + [gamma]
            #Use p value that had highest variance expalined
            #start run timer
            start = timeit.default_timer()
            #gamma = gamma_var[var_cum.index(max(var_cum))]
            gamma = 15
            k = k_trick(x,gamma)
        
        #center grahm matrix
        k_center = AJ_kernels.center_kernel(k)
   
        #run PCA
        k_pca, k_var_explain = AJ_PCA.pca(k_center,n_components=2)
        
        #end and monitor run time to compare performance
        end = timeit.default_timer()
        AJ_run_time = end-start
        dict_name = 'AJ_' + generator.__name__ + '_' + k_trick.__name__
        run_times[dict_name] = AJ_run_time
        
        #Track variance explained
        variance_explained[dict_name] = k_var_explain
                
        #Get non-kernelized PCA
        pca, var_explain = AJ_PCA.pca(x,n_components=2,need_cov=True)
        dict_name = 'AJ_' + generator.__name__ + '_nonkernlized'
        variance_explained[dict_name] = var_explain
        
        #Plot to compare
        AJ_PCA.pca_plots(k_trick,generator,x,y,pca,k_pca,"AJ_")                  
#%%            
#Iterate through kernels to compare with non-kernelized PCA
for k_trick in benchmark_kernels:
    #Iterate through data benchmarks
    for generator in data_benchmarks:
               
        #Generate data
        x, y = generator(n_samples=250, random_state=613)
        
        #Track run time to compare
        start = timeit.default_timer()
        
        #Run scikitlearn's PCA
        SK_k_pca = KernelPCA(n_components=2, kernel=k_trick, 
                             gamma=15).fit(x)
        
        #Record run time
        end = timeit.default_timer()
        SK_run_time = end-start
        dict_name = 'SK_' + generator.__name__ + '_' + k_trick
        run_times[dict_name] =SK_run_time
        
        #Run non kernilized PCA
        SK_pca = PCA(n_components=2).fit(x)
        SK_explain = np.cumsum(SK_pca.explained_variance_ratio_[:2])
        
        #Track explained varience
        n_components = 2
        eig_vals_total = sum(SK_k_pca.lambdas_)
        var_exp = [(i / eig_vals_total)*100 for i in SK_k_pca.lambdas_]
        cum_var_exp = np.cumsum(var_exp)
        cum_var_exp = cum_var_exp[(n_components-1)]
        variance_explained[dict_name] = cum_var_exp 
        
        dict_name = 'SK_' + generator.__name__ + '_nonkernlized'
        variance_explained[dict_name] = SK_explain
        
        SK_pca = PCA(n_components=2).fit_transform(x)
        SK_k_pca = KernelPCA(n_components=2, kernel=k_trick, 
                             gamma=15).fit_transform(x)
        
        #Plot to compare
        AJ_PCA.pca_plots(k_trick,generator,x,y,SK_pca,SK_k_pca,"SK_",True)  
        
#Save results
pd.DataFrame.from_dict(run_times, orient='index').to_csv(
        'outputs/pca/pca_runtimes.csv')
pd.DataFrame.from_dict(variance_explained, orient='index').to_csv(
        'outputs/pca/variance_explained.csv')

#%%
###############################################################################
#Lab section 2
#Compare K-Means with kernelized K-Means
###############################################################################

#Creat list of Kernel tricks to iterate through
kernels = [AJ_kernels.linear_kernel,AJ_kernels.rbf_kernel,
           AJ_kernels.poly_kernel,AJ_kernels.laplacian_kernel]

benchmark_kernels = ["linear","rbf","poly"]

data_benchmarks = [make_moons,make_circles,make_classification]

k_means_acc = {}

#Iterate through kernels to compare with non-kernelized Kmeans
for k_trick in kernels:
    #Iterate through data benchmarks
    for generator in data_benchmarks:
        #Generate data
        x, y = generator(n_samples=250, random_state=613)

        #start run timer
        start = timeit.default_timer()

        #start by kernelizing data
        #Add params when necessary
        if (k_trick.__name__ == "linear_kernel"):
            k = k_trick(x)
        elif (k_trick.__name__ == "poly_kernel"):
            poly = 1
            k = k_trick(x,poly)
            acc = []
            poly_var = []
            for poly in range(1,50):
                k = k_trick(x,poly)
                k_center = AJ_kernels.center_kernel(k)
                #k_centroids,k_clusters = AJ_kmeans.kmeans(k,2)
                k_clusters = KMeans(n_clusters=2).fit_predict(k)
                curr_acc = metrics.accuracy_score(y,k_clusters)
                acc = acc + [curr_acc]
                poly_var = poly_var + [poly]
            #Use p value that had highest variance expalined
            poly = poly_var[acc.index(max(acc))]
            k = k_trick(x,poly)
        else:
            #Iterate through to find gamma with highest accuracy
            acc = []
            gamma_var = []
            for gamma in range(1,50):
                k = k_trick(x,gamma)
                k_center = AJ_kernels.center_kernel(k)
                #k_centroids,k_clusters = AJ_kmeans.kmeans(k,2)
                k_clusters = KMeans(n_clusters=2).fit_predict(k)
                curr_acc = metrics.accuracy_score(y,k_clusters)
                acc = acc + [curr_acc]
                gamma_var = gamma_var + [gamma]
            #Use p value that had highest variance expalined
            gamma = gamma_var[acc.index(max(acc))]
            k = k_trick(x,gamma)

        #non kernilized Kmeans
        centroids,clusters = AJ_kmeans.kmeans(x,2)     
        
        #kernelized Kmeans
        #k_centroids,k_clusters = AJ_kmeans.kmeans(k,2)
        k_clusters = KMeans(n_clusters=2).fit_predict(k)
        
        #Track accuracy
        #kernelized
        k_acc = metrics.accuracy_score(y,k_clusters)
        dict_name = 'AJ_' + generator.__name__ + '_' + k_trick.__name__
        k_means_acc[dict_name] = k_acc
        #non-kernelized
        k_acc = metrics.accuracy_score(y,clusters)
        dict_name = 'AJ_' + generator.__name__ + '_nonkernlized'
        k_means_acc[dict_name] = k_acc
        
        #plot original dataset
        AJ_kmeans.kmeans_plots(x,y,k_trick,generator,clusters,k_clusters)
        
pd.DataFrame.from_dict(k_means_acc, orient='index').to_csv(
        'outputs/kmeans/kmeans_accuracy.csv')

#%%
###############################################################################
#Lab section 3
#Compare Logistic Regression with kernelized Logistic Regression
###############################################################################
#Load the wisconsin breast cancer dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
bc = pd.read_csv(
    filepath_or_buffer=url, 
    header=None, 
    sep=',')

# drops the empty line at file-end
bc.dropna(how="all", inplace=True)

#Split features and label
bc_x = bc.ix[:,2:].values
bc_y = bc.ix[:,1]

#Standardize data in order to get mean=0 and the variance=1
bc_x_std = StandardScaler().fit_transform(bc_x)

#Normal logistic regression
reg_score = cross_val_score(LogisticRegression(), bc_x_std, bc_y, 
                            scoring='accuracy',cv=5)
reg_results = reg_score.mean()
print('Normal log reg results:')
print(reg_results)

#linear kernel logistic regression
bc_linear = AJ_kernels.linear_kernel(bc_x_std)
linear_reg_score = cross_val_score(LogisticRegression(), bc_linear, bc_y, 
                            scoring='accuracy',cv=5)
linear_reg_results = linear_reg_score.mean()
print('Linear kernelized log reg results:')
print(linear_reg_results)

#Gaussian rbf kernel logistic regression
rbf_reg_results = []
for gamma in range(100):
    bc_rbf = AJ_kernels.rbf_kernel(bc_x_std,gamma)
    rbf_reg_score = cross_val_score(LogisticRegression(), bc_rbf, bc_y, 
                                scoring='accuracy',cv=5)
    rbf_reg_results = rbf_reg_results + [rbf_reg_score.mean()]
print('Best Gaussian RBF kernelized log reg results:')
print(max(rbf_reg_results))

#poly kernel logistic regression
poly_reg_results = []
for p in range(10):
    bc_poly = AJ_kernels.poly_kernel(bc_x_std,p)
    poly_reg_score = cross_val_score(LogisticRegression(), bc_poly, bc_y, 
                                scoring='accuracy',cv=5)
    poly_reg_results = poly_reg_results + [poly_reg_score.mean()]
print('Best polynomial kernelized log reg results:')
print(max(poly_reg_results))

#laplacian kernel logistic regression
lap_reg_results = []
for gamma in range(100):
    bc_lap = AJ_kernels.laplacian_kernel(bc_x_std,gamma)
    lap_reg_score = cross_val_score(LogisticRegression(), bc_lap, bc_y,
                                scoring='accuracy',cv=5)
    lap_reg_results = lap_reg_results + [lap_reg_score.mean()]
print('Best laplacian kernelized log reg results:')
print(max(lap_reg_results))
