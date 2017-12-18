# -*- coding: utf-8 -*-
"""
Advance Machine Learning Lab 3
MLDM M2

Subject: Predicting on highly unbalanced datasets.
Dataset is using fraudulant transactions.

Austin Schwinn & Usama Javaid
Dec 15, 2017s
"""
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os

#%%
#Load data
os.chdir('D:/GD/MLDM/Adv Machine Learning/Lab 3/fraud')
x = pd.read_csv('fraud_train_oneday.csv', header=None)
y = pd.read_csv('fraud_train_label_oneday.csv', header=None)

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
#Add weights
weights = y_smote
weights[weights == 1] = 100
weights[weights != 1] = 1

#%%
#Implement resampled dataset with AdaBoostClassifier
ada = AdaBoostClassifier().fit(x_smote,y_smote,weights)

pred_val = ada.predict(x_val)

ada_results = confusion_matrix(y_val,pred_val)
ada_results
