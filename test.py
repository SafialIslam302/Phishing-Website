import math
import tensorflow as tf
from sklearn import tree
from statistics import mean
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

import sys
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.svm import SVC
from keras.models import Sequential
from sklearn.metrics import plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import train_test_split

from keras.layers import Activation,BatchNormalization
from keras.layers.core import Dense,Dropout
from sklearn.metrics import plot_roc_curve
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier




Data = pd.read_csv("PhishingData.csv",header="infer")


x=Data.iloc [:,1:-1]
x=x.values
y=Data.iloc[:,-1].values
dimension = Data.shape[1] - 1
print(dimension)



np.set_printoptions(threshold=sys.maxsize)
y = np.where(y >= 0, 1, -1)



def evaluation(clf, X, Y):
    print(f'Accuracy')
    acc = cross_val_score(clf, X, Y, scoring="accuracy", cv = 5)
    print(acc)
    print("Accuracy Score (Mean): ", acc.mean())
    print("Standard Error: ", acc.std())
    

    print(f'\nF1 Score')
    f1_score = cross_val_score(clf, X, Y, scoring="f1", cv = 5)
    print(f1_score)
    print("F1 Score (Mean): ", f1_score.mean())
    print("Standard Error: ", f1_score.std())
    
    print(f'\nPrecision')
    pre = cross_val_score(clf, X, Y, scoring="precision", cv = 5)
    print(pre)
    print("Precision (Mean): ", pre.mean())
    print("Standard Error: ", pre.std())
    
    print(f'\nSensitivity')
    rec = cross_val_score(clf, X, Y, scoring="recall", cv = 5)
    print(rec)
    print("Recall (Mean): ", rec.mean())
    print("Standard Error: ", rec.std())
    
    
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)

def hyperParameterTuning_DecisionTree(features, labels):
    params = {
        "max_features": ["auto", "sqrt", "log2"],
        "min_samples_leaf": [3, 4, 5],
        "min_samples_split": [8, 10, 12],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, 40, 50],
        "random_state": [10, 20, 30, 40, 50]
    }
    
    rf_model = DecisionTreeClassifier()
    
    gsearch = GridSearchCV(estimator = rf_model, param_grid = params, cv = 5, n_jobs = -1, verbose = 1)
    
    gsearch.fit(features,labels)
    
    return gsearch.best_params_
  
  
hyperParameterTuning_DecisionTree(X_train, y_train)

clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=20, max_features='log2',
                                      min_samples_leaf = 5, min_samples_split = 12, random_state = 10 )

evaluation(clf_tree, X_test, y_test)

