# -*- coding: utf-8 -*-
"""
Implement kmeans for 
advance machine learning lab 1

By: Austin Schwinn, 
Jeremie Blanchard

October 10, 2017
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#create kmeans algorithm
def kmeans(X, k):
    #Iniate atrandom centroild
    centroids = np.random.random((k, X.shape[1]))
    #Store current centroids to compare to new centroids
    prev_centroids = np.zeros(centroids.shape)
    #Cluster labels
    clusters = np.zeros(len(X))
    #Calc distance between new and previous centroids
    dist = np.linalg.norm(centroids - prev_centroids, axis=None)
    #Run until there is no change between centroids of consecutive steps
    print('Before loop')
    count = 0
    while dist != 0:
        count += 1
        #Find closest cluster for each observation
        for i in range(len(X)):
            #Calculate the distnace between each observation and clusts
            clust_dist = np.linalg.norm(X[i] - centroids, axis=1)
            clusters[i]  = np.argmin(clust_dist)
        #Move current centroids to previous centroids matrix
        prev_centroids = centroids.copy()
        #Calculate new centroids by taking average
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            centroids[i] = np.mean(points, axis=0)
        dist = np.linalg.norm(centroids - prev_centroids, axis=None)
        print('Converging step: '+str(count))
    print('centroids converged')
    
    return centroids, clusters

#plot kmeans
def kmeans_plots(x,y,k_trick,generator,clusters,k_clusters):
    #plot original dataset
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], alpha=.75, c=y, 
        cmap=plt.cm.rainbow)
    name = str(generator.__name__) + ' artificial dataset'
    filename = 'outputs/kmeans/' + name
    plt.title(name)
    plt.savefig(filename)
    plt.show()
    plt.close()
    
    #plot non-kernelized kmeans
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], alpha=.75, c=clusters, 
        cmap=plt.cm.rainbow)
    name = str(generator.__name__) + ' non-kernlized k-means'
    filename = 'outputs/kmeans/' + name
    plt.title(name)
    plt.savefig(filename)
    plt.show()
    plt.close()
    
    #plot kernelized kmeans
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], alpha=.75, c=k_clusters, 
        cmap=plt.cm.rainbow)
    name = str(generator.__name__) + ' ' + str(k_trick.__name__) + 'kmeans'
    filename = 'outputs/kmeans/' + name
    plt.title(name)
    plt.savefig(filename)
    plt.show()
    plt.close()