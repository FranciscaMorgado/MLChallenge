# -*- coding: utf-8 -*-
"""
Data Preprocessing
Best Model Search
Save Best Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

PCA_ANALYSIS = False
PCA_ACTIVATED = True
BALANCE = True
TEST_CLASSIFIERS = False
GRID_SEARCH = False


#################### PRE PROCESSING ####################

### READ DATASET ###
df = pd.read_csv('DatasetML.csv')


### SPLIT CATEGORICAL DATA FROM NUMERICAL DATA ###
categorical = []
numerical = []
yy = np.array(df['LABEL'])

for name in df.columns:
    if name=='LABEL':
        continue
    if df[name].dtype=='object':
        categorical.append(df[name].values.tolist())
    else:
        numerical.append(df[name].values.tolist()) 
        
categorical = np.swapaxes(np.array(categorical), 0,1)
numerical = np.swapaxes(np.array(numerical), 0,1)


### CATEGORICAL TO NUMERICAL ###
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(categorical)
categorical_onehot = enc.transform(categorical).toarray()

xx_onehot = np.concatenate((categorical_onehot, numerical), axis=1)
pickle.dump(enc, open('encoder.sav', 'wb'))

### STANDARDIZATION ###
xx_stats = [] # Save statistics to process new data
xx_standardized = xx_onehot.copy()
for i in range(len(xx_onehot[0])):
    col = xx_onehot[:,i]
    col_mean = np.mean(col)
    col_std = np.std(col)
    xx_stats.append([col_mean, col_std]) 
    xx_standardized[:,i] = (col-col_mean)/col_std # Standardize

xx_stats = np.swapaxes(np.array(xx_stats), 0,1)
np.save('data_stats.npy', xx_stats)

### PCA ###
if PCA_ANALYSIS:
    #Finding the best number of components
    pca = PCA().fit(xx_standardized)
    
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') 
    plt.show() #best number of components: 45

if PCA_ACTIVATED:
    pca_best = PCA(n_components=45)
    xx_standardized = pca_best.fit_transform(xx_standardized)
    pickle.dump(pca_best, open('pca.sav', 'wb'))

### SPLIT TRAIN AND TEST ###
xx_train, xx_test, yy_train, yy_test = train_test_split(
        xx_standardized, yy, test_size=0.3, random_state=100)


### IMBALANCED LEARNING ###
if BALANCE:
    smote_enn = SMOTEENN(random_state=0)
    xx_train, yy_train = smote_enn.fit_resample(xx_train, yy_train)



#################### TRAIN AND TEST ####################
    
### TEST DIFFERENT CLASSIFIERS ###
if TEST_CLASSIFIERS:
    header = []
    accuracy = ['accuracy']
    precision = ['precision']
    f1score = ['F1 score']
    auc_list = ['AUC']
    for i in range(7):
        if i==0:
            name = 'RandomForest'
            clf = RandomForestClassifier(
                    n_estimators=25, random_state=0)
        elif i==1:
            name = 'GradientBoosting'
            clf = GradientBoostingClassifier(random_state=0)
        elif i==2:
            name = 'AdaBoost'
            clf = AdaBoostClassifier(random_state=0)
        elif i==3:
            name = 'SupportVectorMachines'
            clf = svm.SVC(random_state=0, probability=True)
        elif i==4:
            name = 'k-NearestNeighbors'
            clf = KNeighborsClassifier()
        elif i==5:
            name = 'LogisticRegression'
            clf = LogisticRegression(random_state=0)
        elif i==6:
            name = 'MultilayerPerceptron'
            clf = MLPClassifier(
                hidden_layer_sizes=(5,2), random_state=0)
        clf.fit(xx_train, yy_train)
        yy_pred = clf.predict(xx_test)
        yy_scores = clf.predict_proba(xx_test)
        
        header.append(name)
        acc = accuracy_score(yy_test, yy_pred)
        accuracy.append(acc)
        prec = precision_score(yy_test, yy_pred)
        precision.append(prec)
        f1 = f1_score(yy_test, yy_pred)
        f1score.append(f1)
        auc = roc_auc_score(yy_test, yy_scores[:,1])
        auc_list.append(auc)
        
    results = np.array([accuracy, precision, f1score, auc_list])
    clf_scores = pd.DataFrame(data=results[:,1:], index=results[:,0], columns=header)
    #Best Classifier: SVM


### GRID SEARCH ###
if GRID_SEARCH:
    parameters = {'kernel':('linear', 'rbf', 'sigmoid', 'poly'),
                  'degree':[3,4,5],
                  'C':[1, 5, 10],
                  'gamma':('auto', 'scale')}
    clf_svm = svm.SVC(random_state=0)
    clf = GridSearchCV(clf_svm, parameters, cv=5)
    clf.fit(xx_train, yy_train)
    
    best_parameters = clf.best_params_ #{'C': 1, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
else:
    best_parameters = {'C': 1, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}

best_clf = clf_svm = svm.SVC(**best_parameters, random_state=0)
best_clf.fit(xx_standardized, yy)

### SAVE MODEL ###
filename = 'model.sav'
pickle.dump(best_clf, open(filename, 'wb'))
 