# -*- coding: utf-8 -*-
"""
Advanced Machine Learning Lab
MLDM M2

Implimentation of Online Passive Aggressive Algorithm

Austin Schwinn
Jeremie Blanchard
Oussama Bouldjedri

"""
#Passive-aggressive
#Binary Classification Algorithm
#%%
import numpy as np
from sklearn.datasets import make_classification,make_moons
import matplotlib.pyplot as plt
from sklearn import svm
import time
import random
#%%
#Avoid overfitting
C= 0.1
accuracy_=np.zeros(1000)
time_tot = 0
X,y = make_classification(n_samples=2000,shuffle = True,random_state=None)

#%% Flip some labels

for t in range(0,100,1):
    a = random.randint(0, len(y)-1)
    b = random.randint(0, len(y)-1)
    c = y[a]
    y[a] = y[b]
    y[b] = c
    

#%%
X_train = X[0:1000,]
X_test = X[1000:2000,]
y = y*2-1
y_train = y[0:1000,]
y_test = y[1000:2000,]
w=np.zeros(len(X[0]))

#%%

for t in range(len(X_train)):
    start=time.time()
    accuracy = 0
    dot_xt = np.dot(w.T,X_train[t])
    pred_y = np.dot(w,X_train[t])
    
    lt = max(0,1 - y_train[t]*dot_xt)
    #tt = lt/np.square(np.linalg.norm(X_train[t],2))
    #tt = min(C,lt/np.square(np.linalg.norm(X[t],2)))
    tt = lt/(np.square(np.linalg.norm(X[t],2))+(1/(2*C)))
    w = w+(tt*y_train[t])*X_train[t]
    end = time.time()
    time_tot += end-start

    for i in range(len(X_test)):
        dot_xt = np.dot(w.T,X_test[i])
        pred_y = np.dot(w,X_test[i])
        #print("pred = ",np.sign((pred_y)))
        #print("value = ",y[t])
        if np.sign(pred_y) == np.sign(y_test[i]):
            accuracy+=1
    accuracy_[t] = accuracy/len(X_test)

    if t%100 == 0:
        print("Accuracy : ",accuracy)
print("Final Accuracy : ",accuracy_[-1])
print("Time : ",time_tot)
    
plt.plot(range(0,1000,1),accuracy_)

#%%

clf = svm.SVC(max_iter = 1)

start=time.time()
clf.fit(X_train, y_train)
end = time.time()
time_tot = end-start

#clf.predict(X_test[1])

accuracy = 0

for i in range(len(X_test)):
    pred_y = clf.predict(X_test[i])
    print("pred_y = ", pred_y)
    if pred_y == y_test[i]:
        accuracy+=1
print(accuracy/len(X_test))
print("time :", time_tot)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    