# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:18:41 2018

@author: Aditya
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def computeStats(series, normalizeCount = 1.0):
    '''
        Compute unique values, high frequency and low frequency statistics for 
        every feature in dataset.
        
        Input
        -----------------------------------------------------------------------
        Series for which stats need to be calculated: series
        Normalize the count value: normalizeCount
        
        Output
        -----------------------------------------------------------------------
        Unique values in a series: nbUnique
        High frequency of specific user for that feature: hiFreq
        Low frequency of a specific user for that feature: loFreq
    '''
    n = float(series.shape[0])
    counts = series.value_counts()
    nbUnique = counts.count() / normalizeCount
    hiFreq = counts[0] / n
    loFreq = counts[-1] / n

    return (nbUnique, loFreq, hiFreq)

def drop_values(new_data,cols):
    '''
        Drop columns which can't be used for training the model.
        
        Input
        -----------------------------------------------------------------------
        DataFrame of features to be used for training: new_data
        
        Output
        -----------------------------------------------------------------------
        Dataframe after dropping values: new_data
    '''
    new_data.drop(cols,axis=1, inplace=True)
    new_data.dropna(inplace=True)
    return new_data


def scaling(X_train,X_test,y_train):
    '''
        Balances unbalanced class by adding synthetic data using SMOTE technique.
        Also, scale and tansform data.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_train
        label for train data : y_train
        Features pertaining to test data : X_test
        
        Output
        -----------------------------------------------------------------------
        New and transformed train data: X_train_t
        Transformed test data: X_test_t
        Labels for new train data: y_train
        
    '''
    
    sm = SMOTE(random_state=12, ratio = 1)
    X_train_res, y_train = sm.fit_sample(X_train, y_train)
    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train_res)
    X_test_t = scaler.transform(X_test)
    return (X_train_t, X_test_t,y_train)


def Gradient_boosting_Classifier(X_tr,X_te, y_tr, y_te, grid_gbc):
    '''
        Predicts on the best Gradient Boosting Classifier after RandomSearch and 
        prints recall and ROC_AUC score.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
        Output
        -----------------------------------------------------------------------
        Returns the best Gradient Boosting Classifier
    '''
    gbc= GradientBoostingClassifier()
    random_search_gbc = RandomizedSearchCV(gbc, grid_gbc, cv=10)
    random_search_gbc.fit(X_tr, y_tr)
    best_random_gbc = random_search_gbc.best_estimator_
    prediction = best_random_gbc.predict(X_te)
    print("Recall For Gradient Boosting is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For Gradient Boosting is: {}".format(roc_auc_score(y_te,prediction)))
    return best_random_gbc


def Random_Forest(X_tr,X_te, y_tr, y_te, grid_rf):
    '''
        Predicts on the best Random Forest Classifier after RandomSearch and 
        prints recall and ROC_AUC score.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
        Output
        -----------------------------------------------------------------------
        Returns the best Random Forest Classifier
    '''
    rf= RandomForestClassifier(class_weight={1:3, 0:1})
    random_search_rf = RandomizedSearchCV(rf, grid_rf, cv=10)
    random_search_rf.fit(X_tr, y_tr)
    best_random_rf = random_search_rf.best_estimator_
    prediction = best_random_rf.predict(X_te)
    print("Recall For Random Forest is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For Random Forest is: {}".format(roc_auc_score(y_te,prediction)))
    return best_random_rf

def Logistic_regression(X_tr,X_te, y_tr, y_te):
    '''
        Predicts on the Logistic Regression Classifier and prints recall 
        and ROC_AUC score.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
    '''
    logreg=LogisticRegression()
    logreg.fit(X_tr, y_tr)
    prediction = logreg.predict(X_te)
    print("Recall For Logistic Regression is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For Logistic Regression is: {}".format(roc_auc_score(y_te,prediction)))
    
    
    

def Support_vector_machine(X_tr,X_te, y_tr, y_te, grid_svc):
    '''
        Predicts on the best SVM Classifier after RandomSearch and 
        prints recall and ROC_AUC score.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
        Output
        -----------------------------------------------------------------------
        Returns the best SVM Classifier
    '''
    svc= SVC()
    random_search_svc = RandomizedSearchCV(svc, grid_svc, cv=10)
    random_search_svc.fit(X_tr, y_tr)
    best_random_svc = random_search_svc.best_estimator_
    prediction = best_random_svc.predict(X_te)
    print("Recall For SVC is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For SVC is: {}".format(roc_auc_score(y_te,prediction)))
    return best_random_svc



def XG_Boost(X_tr,X_te, y_tr, y_te, grid_xg):
    '''
        Predicts on the best XgBoost Classifier after RandomSearch and 
        prints recall and ROC_AUC score.
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
        Output
        -----------------------------------------------------------------------
        Returns the best XGBoost Classifier
    '''
    xg = xgb.sklearn.XGBClassifier()
    random_search_xg = RandomizedSearchCV(xg, grid_xg, cv=10)
    random_search_xg.fit(X_tr, y_tr)
    best_random_xg = random_search_xg.best_estimator_
    prediction = best_random_xg.predict(X_te)
    print("Recall For XG Boosting is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For XG Boosting is: {}".format(roc_auc_score(y_te,prediction)))
    return best_random_xg



def NN(X_tr,X_te, y_tr, y_te):
    '''
        Build a Neural Network with certain given layers and with different 
        activation function. Predict on this network to get recall and
        ROC_AUC score    
        
        Input
        -----------------------------------------------------------------------
        Features pertaining to train data : X_tr
        label for train data : y_tr
        Features pertaining to test data : X_te
        label for test data : y_te
        
    '''
    ohe = LabelEncoder().fit(y_tr)
    y_train = ohe.transform(y_tr)
    y_test = ohe.transform(y_te)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='sigmoid'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(12, activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.fit(X_tr, y_train, epochs=250, batch_size=10)
    
    prediction = model.predict_classes(X_te)
    print("Recall For NN is: {}".format(recall_score(y_te, prediction)))
    print("ROC_AUC For NN is: {}".format(roc_auc_score(y_te,prediction)))