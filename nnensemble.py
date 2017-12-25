# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:40:08 2017

@author: Sider007
"""
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def manipulate_data():
    X1_data = np.random.randint(50,size=250)
    X2_data = np.random.randint(50,size=250)

    X1_b_data = X1_data - np.random.randint(25,size=250)
    X2_b_data = X2_data - np.random.randint(25,size=250)

    X_data = np.vstack((np.hstack((X1_data,X1_b_data)),np.hstack((X2_data,X2_b_data))))
    y_data = np.hstack((np.ones(250),np.zeros(250)))
    y_data = y_data.T.reshape(1,500)
    data = np.vstack((X_data,y_data))
    data = data.T
    values = data
    df = pd.DataFrame(values)
    df.to_csv("dadada.csv", sep=',')
    np.random.shuffle(data)
    X = np.delete(data,1,1)
    y = data[:,-1]
    return data,X,y

def load_train_test_data(X,y,train_ratio=.5):
    
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(np.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=9,n_jobs=-1)
    knn.fit(X_train,y_train)
    
    return knn.predict(X_test), knn.predict_proba(X_test), knn.predict_proba(X_train)
    
def logreg(X_train, y_train, X_test):
    logreg =LogisticRegression()
    logreg.fit(X_train, y_train)
    
    return logreg.predict(X_test), logreg.predict_proba(X_test), logreg.predict_proba(X_train)
    #1/(1+numpy.exp(-(numpy.dot(X_test,logreg.coef_.T))))
    
def dectree(X_train, y_train, X_test):
    dectree = DecisionTreeClassifier(random_state=0)
    dectree.fit(X_train, y_train)

    return  dectree.predict(X_test), dectree.predict_proba(X_test), dectree.predict_proba(X_train)

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
    l1 = add_layer(xs, 2, 10, activation_function=tf.sigmoid)
    prediction = add_layer(l1, 10, 3, activation_function=tf.sigmoid)
    x_data = X.astype(np.float32)
    y_data = y.astype(np.float32)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)))
    train_step = tf.train.AdadeltaOptimizer(0.5).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    
    for i in range(50000):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    
    prediction_value = sess.run(prediction, feed_dict={xs: X_t})        
    return  prediction_value
    

def main(argv):
    data,X,y = manipulate_data()
    X_train, X_test, y_train, y_test = load_train_test_data(X,y,train_ratio=.7)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    
    kn,knp_te,knp_tr=knn(X_train_scale,y_train,X_test_scale)
    knp_te = np.delete(knp_te,1,1)
    knp_tr = np.delete(knp_tr,1,1)
    #print("knn test R^2: %f" % (sklearn.metrics.r2_score(y_test, kn)))
    log,logp_te,logp_tr=logreg(X_train_scale,y_train,X_test_scale)
    logp_te = np.delete(logp_te,1,1)
    logp_tr = np.delete(logp_tr,1,1)
    #print("log test R^2: %f" % (sklearn.metrics.r2_score(y_test, log)))
    dec,decp_te,decp_tr=dectree(X_train_scale,y_train,X_test_scale)
    decp_te = np.delete(decp_te,1,1)
    decp_tr = np.delete(decp_tr,1,1)
    #print("dec test R^2: %f" % (sklearn.metrics.r2_score(y_test, dec)))
    y_1_p_te =np.hstack((np.hstack((knp_te,logp_te)),decp_te)) 
    y_1_p_tr =np.hstack((np.hstack((knp_tr,logp_tr)),decp_tr)) 
    
    yp_tr = nn(X_test_scale,y_1_p_te,X_train_scale)
    print("nn_p test R^2: %f" % (sklearn.metrics.r2_score(yp_tr, y_1_p_tr)))
if __name__ == "__main__":
    main(sys.argv)