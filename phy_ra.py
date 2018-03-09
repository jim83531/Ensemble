# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:51:06 2018

@author: Sider007
"""

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics

def data():
    df=pd.read_csv("minitree_4b_2_26.txt",delimiter=" ",)

    data = df.values
    n,d = data.shape

    X = data[:,2:]
    y_tmp = data[:,:2]
    y = np.zeros((n,1))
    for i in range(n):
        y[i] = y_tmp[i,1]/y_tmp[i,0]

    '''print(data[0])
    print(X[0])
    print(y_tmp[0])
    print(y[0])'''
    
    return X,y


def load_train_test_data(X,y,train_ratio=.5):
    
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(X)
    X_scale = minmax_scaler.transform(X)

    return X_scale

def linreg(X_train, y_train, X_test):
    lin = LinearRegression()
    lin.fit(X_train,y_train)
    
    return lin.predict(X_test)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases 
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
def nn(X,y,X_t):
    xs = tf.placeholder(tf.float32, [None, 17])
    ys = tf.placeholder(tf.float32, [None, 1])
    l1 = add_layer(xs, 17, 25, activation_function = tf.nn.relu)
    l2 = add_layer(l1, 25, 15, activation_function=tf.nn.relu)
    l3 = add_layer(l2, 15, 20, activation_function=tf.nn.relu)
    l4 = add_layer(l3, 20, 35, activation_function=tf.nn.relu)
    l5 = add_layer(l4, 35, 50, activation_function=tf.nn.relu)
    l6 = add_layer(l5, 50, 25, activation_function=tf.nn.relu)
    prediction = add_layer(l6, 25, 1, activation_function= None)
    x_data = X.astype(np.float32)
    y_data = y.astype(np.float32)
    #print(y_data)
    
    loss = tf.losses.mean_squared_error(labels = ys, predictions = prediction)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    
    for i in range(1000):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    prediction_value = sess.run(prediction, feed_dict={xs: X_t})
    sess.close()    
    return  prediction_value





def main(argv):
    X,y = data()
    
    #X_scale = scale_features(X, 0, 1)
    X_train, X_test, y_train, y_test = load_train_test_data(X,y,train_ratio=.7)
    y_hat = nn(X_train,y_train,X_test)
    print(y_hat)
    print(y_test)
    #print("X test R^2: %f" % (sklearn.metrics.r2_score(y_hat, y_test)))
    
    
if __name__ == "__main__":
    main(sys.argv)