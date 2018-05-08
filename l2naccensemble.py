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
    
    '''X1_data = 1 + np.random.uniform(0,10,size=5000)
    X1_b_data = 2 + np.random.uniform(0,10,size=5000)
     
    X2_b_data = 1 + np.random.uniform(0,10,size=5000)
    X2_data = np.ones(5000)
    for i in range(5000):
       for j in range(1,11):
           if (j<=X1_data[i]<j+1):
               X2_data[i] = j + np.random.uniform(0,5)
              
    X3_data = 1 + np.random.uniform(0,10,size=1000)
    X3_b_data = np.ones(1000)
    for i in range(1000):
       for j in range(1,11):
           if (j<=X2_b_data[i]<j+1):
               X3_b_data[i] = j + np.random.uniform(0,5)
    
    a = np.hstack((X1_data,X1_b_data))
    b = np.hstack((X2_data,X2_b_data))
    #c = np.hstack((X3_data,X3_b_data))
    X_data = np.vstack((a,b))

    y_data = np.hstack((np.hstack((np.ones(5000),np.zeros(5000)))))
    y_data = y_data.reshape(1,10000)
    data = np.vstack((X_data, y_data))
    #print(data.shape)
    data = data.T
    
    values = data
    df = pd.DataFrame(values)
    df.to_csv("l2data.csv", sep=',',index = False)'''
    
    df = pd.read_csv('./l2data.csv')
    data = df.values
    
    #np.random.shuffle(data)
    X = np.delete(data,-1,1)
    y = data[:,-1]
    
    '''for i in range (10000):
        if (y[i]==1):
            plt.scatter(X[i,0],X[i,1], c = 'g')
        else:
            plt.scatter(X[i,0],X[i,1], c = 'r')
    plt.show()'''
    
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
            if (abs(a-b)<0.05):
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
def nn(X,X_t,y_1):
    xs = tf.placeholder(tf.float32, [None, 2])
    ys = tf.placeholder(tf.float32, [None, 1])
    l1 = add_layer(xs, 2, 6, activation_function = tf.nn.sigmoid)
    l2 = add_layer(l1, 6, 10, activation_function=tf.nn.relu)
    l3 = add_layer(l2, 10, 5, activation_function=tf.nn.relu)
    '''l4 = add_layer(l2, 10, 5, activation_function=tf.nn.relu)
    l5 = add_layer(l2, 10, 5, activation_function=tf.nn.relu)'''
    prediction_1 = add_layer(l3, 5, 1, activation_function= None)
    '''prediction_2 = add_layer(l4, 5, 1, activation_function= None)
    prediction_3 = add_layer(l5, 5, 1, activation_function= None)'''
    x_data = X.astype(np.float32)
    y_data_1 = y_1.astype(np.float32)
    '''y_data_2 = y_2.astype(np.float32)
    y_data_3 = y_3.astype(np.float32)'''
    #print(y_data)
    
    loss_1 = tf.losses.mean_squared_error(ys, prediction_1)
    '''loss_2 = tf.losses.mean_squared_error(ys, prediction_2)
    loss_3 = tf.losses.mean_squared_error(ys, prediction_3)'''
    train_step = tf.train.AdadeltaOptimizer(0.25).minimize(loss_1)
    '''train_step_2 = tf.train.AdadeltaOptimizer(0.45).minimize(loss_2)
    train_step_3 = tf.train.AdadeltaOptimizer(0.45).minimize(loss_3)'''
    #total_train = tf.train.AdadeltaOptimizer(0.05).minimize(loss_3+loss_2+loss_1)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    n,d = X.shape
    batch_size = 100
    for epoch in range(1000): 
        for start in range(0,n,batch_size):
            end = start + batch_size
            sess.run(train_step, feed_dict={xs: x_data[start:end], ys: y_data_1[start:end]})
            '''sess.run(train_step_2, feed_dict={xs: x_data[start:end], ys: y_data_2[start:end]})
            sess.run(train_step_3, feed_dict={xs: x_data[start:end], ys: y_data_3[start:end]})'''
            # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
            #sess.run(train_step, feed_dict={xs: batch_x, ys: batch_y})
            '''sess.run(train_step_2, feed_dict={xs: x_data, ys: y_data_2})
            sess.run(train_step_3, feed_dict={xs: x_data, ys: y_data_3})'''
        
            '''if(i%3 == 0):
                sess.run(train_step, feed_dict={xs: x_data, ys: y_data_1})
            if(i%3 == 1):
            sess.run(train_step_2, feed_dict={xs: x_data, ys: y_data_2})
            if(i%3 == 2):
            sess.run(train_step_3, feed_dict={xs: x_data, ys: y_data_3})'''
            #print(sess.run(loss_1, feed_dict={xs: batch_x, ys: batch_y}))
            #print(sess.run(loss_2, feed_dict={xs: x_data, ys: y_data_2}))
    
    
    print(sess.run(loss_1, feed_dict={xs: x_data, ys: y_data_1}))
    '''print(sess.run(loss_2, feed_dict={xs: x_data, ys: y_data_2}))
    print(sess.run(loss_3, feed_dict={xs: x_data, ys: y_data_3}))'''
    prediction_value = sess.run(prediction_1, feed_dict={xs: X_t})
    '''prediction_value_2 = sess.run(prediction_2, feed_dict={xs: X_t})
    prediction_value_3 = sess.run(prediction_3, feed_dict={xs: X_t})'''
    #sess.close()
    return  prediction_value#, prediction_value_2, prediction_value_3

def mv(kn,log,dec):
    
    n,d = kn.shape
    mvm = np.zeros((n,1))
    zcount = np.zeros((n,1))
    ocount = np.zeros((n,1))
    for i in range(n) :
        if (kn[i]==0):
            zcount[i] += 1
        else:
            ocount[i] += 1
        if (log[i]==0):
            zcount[i] += 1
        else:
            ocount[i] += 1
        if (dec[i]==0):
            zcount[i] += 1
        else:
            ocount[i] += 1    
        
        if (zcount[i] > ocount[i]):
            mvm[i]=0
        else:
            mvm[i]=1
    
    return mvm

def en(nnyp,my):
    

    
    n, d = my.shape
    en_p = np.zeros((n,1))
    w = np.zeros((1,3))
    en = np.zeros((n,1))

    
    for i in range(n):
        summ = np.sum(nnyp[i])
        w = (nnyp)/summ
        en_p[i] = np.dot(w[i].T.reshape(1,3),my[i].reshape(3,1))
        
        if (en_p[i]>0.5):
            en[i]=1
        else:
            en[i]=0
    
    return en, en_p
    
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
    y_te_pre = np.vstack((np.vstack((knn_prte,svc_prte)),dec_prte))
    y_te_pre = y_te_pre.reshape(2000,3)
    
    
    #caculate exacally accuracy
    knn_exac = cacuexacc(knn_prtr_n, X_train_n, y_train_n)
    svc_exac = cacuexacc(svc_prtr_n, X_train_n, y_train_n)
    dec_exac = cacuexacc(dec_prtr_n, X_train_n, y_train_n)
    y_train_n_exac = np.hstack((np.hstack((knn_exac,svc_exac)),dec_exac))
    
    
    knn_te_exac = cacuexacc(knn_prte, X_test, y_test)
    svc_te_exac = cacuexacc(svc_prte, X_test, y_test)
    dec_te_exac = cacuexacc(dec_prte, X_test, y_test)
    y_test_exac = np.hstack((np.hstack((knn_te_exac,svc_te_exac)),dec_te_exac))
    
    
    
    knn_nn_test = nn(X_train_n,X_test,knn_exac)
    svc_nn_test = nn(X_train_n,X_test,svc_exac)
    dec_nn_test = nn(X_train_n,X_test,dec_exac)
    #y_train_n_nn_test = nn(X_train_n,X_test,y_train_n_exac)
    
    
    #knn_nn_test, svc_nn_test, dec_nn_test= nn(X_train_n,X_test,knn_exac,svc_exac,dec_exac)
    #y_train_n_nn_test = np.hstack((np.hstack((knn_nn_test,svc_nn_test)),dec_nn_test))
    df = pd.read_csv('./prea_exac.csv')
    y_train_n_nn_test = df.values
    
    mvm = mv(knn_prte,svc_prte,dec_prte)
    final_pre, mulre = en(y_train_n_nn_test,y_te_pre)   
    
    
    print(y_te_pre[0:10])
    print(y_train_n_nn_test[0:10])
    print(final_pre[0:10])
    print(mulre[0:10])
    print(y_test[0:10])
    #print(y_test_exac[50:75])
    #print(svc_nn_test[50:75])
    #print(dec_nn_test[50:75])
    #print(y_train_n_nn_test[50:75])
    '''print(svc_exac)
    print(svc_nn_test)
    print(dec_exac)
    print(dec_nn_test)'''
    print("knn nn_p test R^2: %f" % (sklearn.metrics.r2_score(knn_nn_test, knn_te_exac)))
    print("svc nn_p test R^2: %f" % (sklearn.metrics.r2_score(svc_nn_test, svc_te_exac)))
    print("dec nn_p test R^2: %f" % (sklearn.metrics.r2_score(dec_nn_test, dec_te_exac)))
    ##print("nn_p test R^2: %f" % (sklearn.metrics.r2_score(y_train_n_nn_test, y_test_exac)))
    
    print("knn test : %f" % (sklearn.metrics.accuracy_score(knn_prte, y_test)))
    print("svc test : %f" % (sklearn.metrics.accuracy_score(svc_prte, y_test)))
    print("dec test : %f" % (sklearn.metrics.accuracy_score(dec_prte, y_test)))
    print("pre test : %f" % (sklearn.metrics.accuracy_score(mvm, y_test)))
    print("pre test : %f" % (sklearn.metrics.accuracy_score(final_pre, y_test)))
    
    values = y_train_n_nn_test
    df = pd.DataFrame(values)
    df.to_csv("prea_exac.csv", sep=',',index = False)
    '''print("dec nn_p test R^2: %f" % (sklearn.metrics.r2_score(y_train_n_nn_test, y_test_exac)))'''
    
main()