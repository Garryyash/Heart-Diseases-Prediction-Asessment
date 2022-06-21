# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:25:10 2022

@author: caron
"""

#%% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



#%% Static
CSV_PATH = os.path.join(os.getcwd(),('heart.csv'))


#%% Funtions
def plot_con(df,continous_col):
    
    for con in continous_col:
        plt.figure()
        sns.distplot(df[con])
        plt.show()
        
def plot_cat(df,categorical_col):
    
    for cat in categorical_col:
        plt.figure()
        sns.countplot(df[cat])
        plt.show()
#%% Performing EDA

# Step 1) Data loading
df = pd.read_csv(CSV_PATH)


# Step 2) Data inspection & Visualization

#Data inspection
df.info()
temp = df.describe().T

# Visualization

df.boxplot()
plt.show

# plot continuos data

con_columns = ['age','trtbps', 'chol', 'thalachh', 'oldpeak']
plot_con(df, con_columns)

# plot categorical data

cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output']      
plot_cat(df, cat_columns)       


# Check for NaNs
df.isna().sum()  # Number of NaNs
# Check for duplicated samples
df.duplicated().sum() # 1 duplicated sample found


# Step 3) Data Cleaning
# No NaNs is found to filter
df = df.drop_duplicates() # 1 duplicated sample dropped
     
# Data imputation using Simple Imputer
df['thall']=df['thall'].replace(0,np.nan) # 0 is replaced with NaNs
df.isna().sum() # shows 2 NaNs
df['thall'].fillna(df['thall'].mode()[0], inplace=True) # NaNs replaced with mode
df.isna().sum() # shows 0 Nans
    
#%% Step 4) Features Selection 
# No label encoding needed as all samples are numerically labelled and classified


# defining the features(X) and target(y)
X = df.drop(labels=['output'],axis=1)
y = df['output']



# finding Correlation & relationship between cont Vs Cat With Logistic Regresion

for i in X.columns:
    print(i)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(np.expand_dims(X[i],axis=-1),y)
    print(lr.score(np.expand_dims(X[i],axis=-1),y))



#%% Step 5) Data Preprocessing
# there is no NaNs
# Everything in numbers
# Selecting all labels as all the labels has an accuracy more than 50%
X.head()
X.info()
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=7)




#%% Pipeline
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

# Logistic regression pipeline
pl_std_lr = Pipeline([('StandardScaler', StandardScaler()),('LogisticClassifier', LogisticRegression())])

pl_mms_lr = Pipeline([('MinMaxScaler', MinMaxScaler()),('LogisticClassifier', LogisticRegression())])

# KNN pipeline
pl_std_knn = Pipeline([('StandardScaler', StandardScaler()),('KNNClassifier', KNeighborsClassifier())])

pl_mms_knn = Pipeline([('MinMaxScaler', MinMaxScaler()),('KNNClassifier', KNeighborsClassifier())])

# RF pipeline
pl_std_rf = Pipeline([('StandardScaler', StandardScaler()),('RFClassifier', RandomForestClassifier())])

pl_mms_rf = Pipeline([('MinMaxScaler', MinMaxScaler()),('RFClassifier', RandomForestClassifier())])

# DT pipeline
pl_std_dt = Pipeline([('StandardScaler', StandardScaler()),('DTClassifier', DecisionTreeClassifier())])

pl_mms_dt = Pipeline([('MinMaxScaler', MinMaxScaler()),('DTClassifier', DecisionTreeClassifier())])

# SVC pipeline
pl_std_svc = Pipeline([('StandardScaler', StandardScaler()),('SVClassifier', SVC())])

pl_mms_svc = Pipeline([('MinMaxScaler', MinMaxScaler()),('SVClassifier', SVC())])

pipelines = [pl_std_lr,pl_mms_lr,pl_std_knn,pl_mms_knn,pl_std_rf,pl_mms_rf,pl_std_dt,pl_mms_dt,pl_std_svc,pl_mms_svc]

for pipeline in pipelines:
    pipeline.fit(X_train,y_train)
#%% Pipeline analysis
best_accuracy = 0
pipeline_scored = []

for i, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    pipeline_scored.append(pipeline.score(X_test,y_test))

best_pipeline = pipelines[np.argmax(pipeline_scored)]
best_accuracy = pipeline_scored[np.argmax(pipeline_scored)]
print('The best combination of the pipeline is {} with accuracy of {}'.format(best_pipeline.steps,best_accuracy))

# Grid search CV
# From pipeline above, It is deduced that the SS + lr
# Achieved the 85% accuracy when tested against test dataset

pl_std_lr = Pipeline([('StandardScaler', StandardScaler()),
                      ('LogisticClassifier', LogisticRegression())])

from sklearn.model_selection import GridSearchCV

# grid_param(eter)
grid_param = [{'LogisticClassifier__random_state':[100,1000,None],
               'LogisticClassifier__C':[0.001,0.01,0.1,1,10]}]
 
grid_search = GridSearchCV(pl_std_lr,grid_param,cv=5,n_jobs=2)
best_model = grid_search.fit(X_train,y_train)

#%%
# retrain your model the selected parameters

pl_std_lr = Pipeline([('StandardScaler', StandardScaler()),
                      ('LogisticClassifier', LogisticRegression(C=0.1,random_state=100))])

pl_std_rf.fit(X_train,y_train)

BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')
with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(pl_std_rf,file)

#%%
print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)


BEST_PIPE_PATH = os.path.join(os.getcwd(),'best_pipe.pkl')
with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

    
#%% Model Analysis

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

y_true = y_test
y_pred = best_model.predict(X_test)


print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
