# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:41:16 2018

@author: Sid007
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.metrics import brier_score_loss
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from numpy import linalg as LA

def manipulate_data():
    
    X1_data = 1 + np.random.uniform(0,10,size=1000)
    X1_b_data = 2 + np.random.uniform(0,10,size=1000)
     
    X2_b_data = 1 + np.random.uniform(0,10,size=1000)
    X2_data = np.ones(1000)
    for i in range(1000):
       for j in range(1,11):
           if (j<=X1_data[i]<j+1):
               X2_data[i] = j + np.random.uniform(0,5)
    
    a = np.hstack((X1_data,X1_b_data))
    b = np.hstack((X2_data,X2_b_data))
    
    X_data = np.vstack((a,b))

    y_data = np.hstack((np.hstack((np.ones(1000),np.zeros(1000)))))
    y_data = y_data.reshape(1,2000)
    data = np.vstack((X_data, y_data))
    #print(data.shape)
    data = data.T
    
    '''values = data
    df = pd.DataFrame(values)
    df.to_csv("dadada.csv", sep=',',index = False)
    
    df = pd.read_csv('./dadada.csv')
    data = df.values'''
    
    np.random.shuffle(data)
    X = np.delete(data,-1,1)
    y = data[:,-1]
    
    for i in range (2000):
        if (y[i]==1):
            plt.scatter(X[i,0],X[i,1], c = 'g')
        else:
            plt.scatter(X[i,0],X[i,1], c = 'r')
    plt.show()
    
    return data,X,y
def load_train_test_data(X,y,train_ratio=.5):
    
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(X)
    X_scale = minmax_scaler.transform(X)

    return X_scale

def knn(X_train_c, y_train_c, X_train_n,X_test):
    knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    knn.fit(X_train_c,y_train_c)
    
    return knn.predict(X_train_n), knn.predict(X_test)

def svc(X_train_c, y_train_c, X_train_n,X_test):
    svc =SVC(kernel='rbf',max_iter=-1, random_state=None, shrinking=False)
    svc.fit(X_train_c, y_train_c)
    
    return svc.predict(X_train_n), svc.predict(X_test)

def dectree(X_train_c, y_train_c, X_train_n,X_test):
    dectree = DecisionTreeClassifier(random_state=0)
    dectree.fit(X_train_c, y_train_c)

    return  dectree.predict(X_train_n), dectree.predict(X_test)

def cacuexacc(class_prtr_n, X_train_n, y_train_n):
    n,d = X_train_n.shape
    exac = np.zeros((n,1))
    
    for i in range (n):
        cor = 0
        total = 0
        for j in range(n):
            a = LA.norm(X_train_n[i])
            b = LA.norm(X_train_n[j])
            if (abs(a-b)<0.01):
                if(class_prtr_n[j] == y_train_n[j]):
                    cor = cor + 1
                    total = total +1
                else:
                    total = total +1

        acc = cor/total
        exac[i] = acc
        
    #print(exac.shape)
    return exac

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases 
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
def nn(X,y,X_t):
    xs = tf.placeholder(tf.float32, [None, 2])
    ys = tf.placeholder(tf.float32, [None, 3])
    l1 = add_layer(xs, 2, 6, activation_function = tf.nn.sigmoid)
    l2 = add_layer(l1, 6, 15, activation_function=tf.nn.relu)
    l3 = add_layer(l2, 15, 12, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l3, 12, 3, activation_function= None)
    x_data = X.astype(np.float32)
    y_data = y.astype(np.float32)
    #print(y_data)
    
    loss = tf.losses.mean_squared_error(ys, prediction)
    train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    
    for i in range(10000):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    prediction_value = sess.run(prediction, feed_dict={xs: X_t})
    #sess.close()
    return  prediction_value


def main ():
    #資料建構
    data,X,y = manipulate_data()
    X_scale = scale_features(X, 0, 1)
    X_train, X_test, y_train, y_test = load_train_test_data(X_scale,y,train_ratio=.8)
    X_train_c,X_train_n, y_train_c, y_train_n = load_train_test_data(X_train,y_train,train_ratio=.5)
    
    #ensemble members
    knn_prtr_n, knn_prte = knn(X_train_c, y_train_c, X_train_n,X_test)
    svc_prtr_n, svc_prte = svc(X_train_c, y_train_c, X_train_n,X_test)
    dec_prtr_n, dec_prte = dectree(X_train_c, y_train_c, X_train_n,X_test)
    
    #caculate exacally accuracy
    knn_exac = cacuexacc(knn_prtr_n, X_train_n, y_train_n)
    svc_exac = cacuexacc(svc_prtr_n, X_train_n, y_train_n)
    dec_exac = cacuexacc(dec_prtr_n, X_train_n, y_train_n)
    
    y_train_n_exac = np.hstack((np.hstack((knn_exac,svc_exac)),dec_exac))
    
    nn_y_test = nn(X_train_n,y_train_n_exac,X_test)
    print(y_train_n_exac)
    print(nn_y_test)
    
main()