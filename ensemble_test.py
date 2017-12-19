# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:29:01 2017

@author: Sider007
"""

import sys

import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def load_train_test_data(train_ratio=.5):
    df = pandas.read_csv('./creditcard.csv')
    data = df.values
    X = numpy.delete(data,30,1)
    X = numpy.concatenate((numpy.ones((len(X),1)),X), axis=1)
    y = data[:,-1].reshape(-1,1)
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def knn(X_train, y_train, x):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    
    return knn.predict_proba(x),knn.predict(x)
    
def logreg(X_train, y_train, x):
    logreg =LinearSVC(C=1.0, max_iter=1000, penalty='l2', random_state=0, tol=0.0001 )
    logreg.fit(X_train, y_train)

    return 1/(1+numpy.exp(-(numpy.dot(x,logreg.coef_.T)))) , logreg.predict(x)
    
def dectree(X_train, y_train, x):
    dectree = DecisionTreeClassifier(random_state=0)
    dectree.fit(X_train, y_train)

    return dectree.predict_proba(x), dectree.predict(x)

def ensemble_feaure_loss(X_train,y_train,X_test,y_test):
    n,d = X_train.shape
    m,dd = X_test.shape
    for i in range (0,m):
        x = X_test[i].T.reshape(1,d)
        a = numpy.zeros((1,2))
        enp = numpy.zeros((m,1))
        a,a_out = knn(X_train,y_train,x)
        b,b_out = logreg(X_train,y_train,x)
        c,c_out = dectree(X_train,y_train,x)
        if (y_test[i] == 0):
            a_loss = a[0,0]
            b_loss = (1-b)
            c_loss = c[0,0]
        if (y_test[i] == 1):
            a_loss = a[0,1]
            b_loss = b
            c_loss = c[0,1]
        wa = a_loss
        wb = b_loss
        wc = c_loss 
        z = wa + wb + wc
        enp[i] = wa/z*a_out + wb/z*b_out + wc/z*c_out
        print(i)
    return enp

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.8)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    ans = ensemble_feaure_loss(X_train,y_train,X_test,y_test)
    print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, ans)))
    

if __name__ == "__main__":
    main(sys.argv)