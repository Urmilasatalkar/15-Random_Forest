# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:16:09 2024

@author: urmii
"""
'''
2.	 Divide the diabetes data into train and test datasets and build 
a Random Forest and Decision Tree model with Outcome as 
the output variable. 
The business objective appears to involve leveraging data 
related to diabetes to develop predictive models. These models are likely 
intended to aid in medical diagnosis, treatment planning, or risk assessment 
for patients with diabetes.

Maximize:
    1. Sensitivity (True Positive Rate)
    2. Accuracy
    
Minimize:
    1.Prediction time
    2. FP and FN
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('c:/10-ML/RandomForest/Diabetes.csv')
data
data.columns

# Separate features (X) and target variable (y)
X = data.drop(' Class variable', axis=1)  # Features
y = data[' Class variable']  # Target variable
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

model.fit(X_train,y_train)
model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


