# -*- coding: utf-8 -*-
"""
Advance Machine Learning Lab 3
MLDM M2

Subject: Predicting on highly unbalanced datasets.
Dataset using fraudulant transactions. Comparing results
between adaboost and random forest algorithms 

Austin Schwinn & Usama Javaid
Dec 15, 2017s
"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#%%
#Load data
os.chdir('D:/GD/MLDM/Adv Machine Learning/Lab 3/fraud')
x = pd.read_csv('fraud_train_oneday.csv', header=None)
y = pd.read_csv('fraud_train_label_oneday.csv', header=None)
x_test = pd.read_csv('fraud_test.csv',header=None)

#%%
#Split into train and validation set
x,x_val,y,y_val = train_test_split(x,y,train_size=.9)

#%%
#Select just fraudulant transactions
fraud = x[y[0]==1]

#Take smaller sub-sample of non_fraudulant examples
x_sub = x.loc[np.random.choice(x[y[0]!=1].index.values,100000,replace=False),:]

#Add fraudulant transactions back in
x_sub = pd.concat([x_sub,fraud])

#Get subset of labels
y_sub = y.loc[x_sub.index,0]

#%%
#Use smote to oversample
sm = SMOTE(random_state=10)
x_smote, y_smote = sm.fit_sample(x_sub,y_sub)

#%%
#Iterate through different weights to find optimal recall
ada_confusion = []
ada_precision = []
ada_recall = []
rf_confusion = []
rf_precision = []
rf_recall = []

#%%
for i in range(1,11):
    #Add weights
    weights = np.copy(y_smote)
    weights[weights == 1] = i
    weights[weights == 0] = 1
    
    
    #Implement AdaBoostClassifier with balanced dataset
    ada_pred = AdaBoostClassifier(n_estimators=100,random_state=0).fit(x_smote,
                                 y_smote,weights).predict(x_val)

    #Adaboost results
    ada_confusion = ada_confusion + [confusion_matrix(y_val,ada_pred)]
    ada_precision = ada_precision + [precision_score(y_val,ada_pred)]
    ada_recall = ada_recall + [recall_score(y_val, ada_pred)]
    '''
    print(ada_confusion)
    print(ada_precision)
    print(ada_recall)
    '''
    print('ada: ' + str(i))
    
    #Impliment random forestwith balaned dataset
    rf_pred = RandomForestClassifier(n_estimators=100,max_depth=7,
                random_state=0).fit(x_smote, y_smote,weights).predict(x_val)
    
    #Random Forest Results
    rf_confusion = rf_confusion + [confusion_matrix(y_val,rf_pred)]
    rf_precision = rf_precision + [precision_score(y_val,rf_pred)]
    rf_recall = rf_recall + [recall_score(y_val,rf_pred)]
    '''
    print(rf_confusion)
    print(rf_precision)
    print(rf_recall)
    '''
    print('rf: ' + str(i))

#%%
#Plot and compare results

#Adaboost
#Precision vs recall
plt.plot(ada_precision,ada_recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Adaboost Precision vs Recall')
plt.grid(True)
weights = range(1,11)
for label, x, y in zip(weights, ada_precision, ada_recall):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#True positive vs false positive
ada_tp = []
ada_fp = []
for i in range(len(ada_confusion)):
    ada_fp = ada_fp + [ada_confusion[i][0,1]]
    ada_tp = ada_tp + [ada_confusion[i][1,1]]

plt.plot(ada_tp,ada_fp)
plt.xlabel('True Positives')
plt.ylabel('False Positives')
plt.title('Adaboost True vs False Positives')
plt.grid(True)
for label, x, y in zip(weights, ada_tp, ada_fp):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

#Random Forest
#Precision vs recall
plt.plot(rf_precision,rf_recall)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Random Forest Precision vs Recall')
plt.grid(True)
for label, x, y in zip(weights, rf_precision,rf_recall):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
plt.show()

#True positive vs false positive
rf_tp = []
rf_fp = []
for i in range(len(ada_confusion)):
    rf_fp = rf_fp + [rf_confusion[i][0,1]]
    rf_tp = rf_tp + [rf_confusion[i][1,1]]
plt.plot(rf_tp,rf_fp)
plt.xlabel('True Positives')
plt.ylabel('False Positives')
plt.title('Random Forest True vs False Positives')
plt.grid(True)
for label, x, y in zip(weights, rf_tp, rf_fp):
    plt.annotate(
        label,
        xy=(x, y), xytext=(5, 5),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
plt.show()

#%%
#Predict on test set using optimal parameters
weights = np.copy(y_smote)
weights[weights == 1] = 2
weights[weights == 0] = 1

ada_test = AdaBoostClassifier(n_estimators=100,random_state=0).fit(x_smote,
                                 y_smote,weights).predict(x_test)

rf_test = RandomForestClassifier(n_estimators=100,max_depth=7,
            random_state=0).fit(x_smote, y_smote).predict(x_test)

pd.DataFrame(ada_test).to_csv('Javaid_Schwinn_fraud_test_labels_ADABoost.csv',
            index=False)
pd.DataFrame(rf_test).to_csv('Javaid_Schwinn_fraud_test_labels_RandomForest.csv',
            index=False)