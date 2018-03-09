# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:40:08 2017

@author: Sider007
"""
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import pandas as pd
import sklearn.metrics
from sklearn.metrics import brier_score_loss
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets.samples_generator import make_swiss_roll


def manipulate_data():
    
    X1_data = 1 + np.random.randint(10,size=1000)
    #X11_data = np.random.randint(60,size=250)
    X1_b_data = 1 + np.random.randint(10,size=1000)
    #X11_b_data = X11_data + np.random.randint(30,size=250)
    
    #X2_data = 1 + np.random.randint(10,size=1000)
    X2_b_data = 1 + np.random.randint(10,size=1000)
    
    #斜線型
    X2_data = np.ones(1000)
    for i in range(1000):
        for j in range(1,10):
            if (j<=X1_data[i]<j+1):
                X2_data[i] = j + np.random.randint(5)
    
    '''
    X2_b_data = np.ones(1000)
    for i in range(1000):
        for j in range(1,10):
            if (j<=X1_data[i]<j+1):
                X2_data[i] = j + np.random.randint(5)
    '''      
    #X3_data = 1 + np.random.randint(5,size=1000)
    X3_data = np.ones(1000)
    for i in range(1000):
        if (1<=X1_data[i]<5):
                X3_data[i] = (X1_data[i] * X2_data[i]*5 - X2_data[i]/X1_data[i]*2.1)/10 - np.random.randint(5)
        if (5<=X1_data[i]<11):
                X3_data[i] = (X1_data[i] * X1_data[i]*2.3 - X1_data[i])/10 - np.random.randint(7)
    #X3_b_data = 1 + np.random.randint(10,size=1000)
    X3_b_data = np.ones(1000)
    for i in range(1000):
        if (1<=X1_b_data[i]<3):
                X3_b_data[i] = (X2_b_data[i] * X2_b_data[i]*1.6 + X1_b_data[i]*X1_b_data[i]*0.2)/10 - np.random.randint(5)
        if (3<=X1_b_data[i]<11):
                X3_b_data[i] = (X1_b_data[i] * X1_b_data[i]*2.3 - X1_data[i]*X2_b_data[i]*0.5)/10 - np.random.randint(3)
    #X3_data = (X1_data * 3 + X1_data * X1_data)/13
    #X3_b_data = (X2_data * 5+ X2_data * X2_data * 4 + X1_data / X2_data * 2.7)/70
    '''
    X4_data = X1_data / X2_data + np.random.randint(10,size=1000)
    X4_b_data = X1_data * X2_data + np.random.randint(10,size=1000)
    
    X5_data = X1_data * 3 + X1_data * X1_data + np.random.randint(10,size=1000)
    X5_b_data = X2_data * 5+ X2_data * X2_data * 4 + np.random.randint(10,size=1000)
    '''
    
    '''
    #斜線型
    X2_data = np.zeros(1000)
    for i in range(1000):
        for j in range(10,40):
            if (j<=X1_data[i]<j+1):
                X2_data[i] = j + np.random.randint(15)
        

        
    #X22_data = np.random.randint(60,size=250)
    X2_b_data = 15 + np.random.randint(35,size=1000)
    X2_b_data = np.zeros(1000)
    for i in range(1000):
        for j in range(-10,20):
            if (j<=X1_b_data[i]<j+1):
                X2_b_data[i] =  j + np.random.randint(15)'''
    #X22_b_data = X22_data + np.random.randint(30,size=250)


    a = np.hstack((X1_data,X1_b_data))
    b = np.hstack((X2_data,X2_b_data))
    c = np.hstack((X3_data,X3_b_data))
    #d = np.hstack((X4_data,X4_b_data))
    #e = np.hstack((X5_data,X5_b_data))
    ab = np.vstack((a,b))
    #cd = np.vstack((c,d))
    #x = np.vstack((ab,cd))
    X_data = np.vstack((ab,c))
    

    print(X_data)
    '''n_samples = 1000
    t_0 = 1.5 * np.pi * (1 + 1.5 * np.random.rand(1, n_samples))
    t_1 = 1.5 * np.pi * (2.5 + 1.5 * np.random.rand(1, n_samples))
    x_0 = t_0 * np.cos(t_0)
    y_0 = t_0 * np.sin(t_0)
    X_0 = np.concatenate((x_0, y_0))
    X_0 += .7 * np.random.randn(2, n_samples)
    x_1 = t_1 * np.cos(t_1)
    y_1 = t_1 * np.sin(t_1)
    X_1 = np.concatenate((x_1, y_1))
    X_1 += .7 * np.random.randn(2, n_samples)
    
    X = np.hstack((X_0, X_1))

    X_data = X.T'''

    y_data = np.hstack((np.hstack((np.ones(1000),np.zeros(1000)))))


    y_data = y_data.reshape(1,2000)
    #print(X_data.shape)
    #print(y_data.shape)
    
    data = np.vstack((X_data, y_data))
    #print(data.shape)
    data = data.T


    '''values = data
    df = pd.DataFrame(values)
    df.to_csv("dadada.csv", sep=',')
    
    df = pd.read_csv('./dadada.csv')
    data = df.values'''
    np.random.shuffle(data)
    X = np.delete(data,-1,1)
    y = data[:,-1]
    print(X)
    print(y)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    for i in range(2000):
        if (y[i] ==1):
            ax.scatter(X[i,0],X[i,1],X[i,2], c = 'g')
        else:
            ax.scatter(X[i,0],X[i,1],X[i,2], c = 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return data,X,y

def load_train_test_data(X,y,train_ratio=.5):
    
    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(X)
    X_scale = minmax_scaler.transform(X)

    return X_scale

def knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    knn.fit(X_train,y_train)
    
    return knn.predict(X_test), knn.predict_proba(X_train), knn.predict_proba(X_test)
    
def logreg(X_train, y_train, X_test):
    logreg =LogisticRegression()
    logreg.fit(X_train, y_train)
    
    return logreg.predict(X_test), logreg.predict_proba(X_train), logreg.predict_proba(X_test)
    #1/(1+numpy.exp(-(numpy.dot(X_test,logreg.coef_.T))))
    
def dectree(X_train, y_train, X_test):
    dectree = DecisionTreeClassifier(random_state=0)
    dectree.fit(X_train, y_train)

    return  dectree.predict(X_test), dectree.predict_proba(X_train), dectree.predict_proba(X_test)

def cali_knn(X_train, y_train, X_test):
    calknn = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=5,n_jobs=-1),cv=2,method='sigmoid')
    calknn.fit(X_train,y_train)
    return calknn.predict(X_test), calknn.predict_proba(X_train), calknn.predict_proba(X_test)

def cali_logreg(X_train, y_train, X_test):
    callog = CalibratedClassifierCV(LogisticRegression(),cv=2,method='sigmoid')
    callog.fit(X_train,y_train)
    return callog.predict(X_test), callog.predict_proba(X_train), callog.predict_proba(X_test)

def cali_dectree(X_train, y_train, X_test):
    callog = CalibratedClassifierCV(DecisionTreeClassifier(),cv=2,method='sigmoid')
    callog.fit(X_train,y_train)
    return callog.predict(X_test), callog.predict_proba(X_train), callog.predict_proba(X_test)

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
    xs = tf.placeholder(tf.float32, [None, 3])
    ys = tf.placeholder(tf.float32, [None, 3])
    l1 = add_layer(xs, 3, 25, activation_function = tf.nn.sigmoid)
    #l2 = add_layer(l1, 25, 30, activation_function=tf.nn.relu)
    #l3 = add_layer(l2, 30, 15, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l1, 25, 3, activation_function= None)
    x_data = X.astype(np.float32)
    y_data = y.astype(np.float32)
    #print(y_data)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction)))
    train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    
    
    for i in range(30000):
        # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    prediction_value = sess.run(prediction, feed_dict={xs: X_t})
    #sess.close()    
    return  abs(prediction_value)
    
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
    

    
    n,d =my.shape
    en_p = np.zeros((n,1))
    w = np.zeros((1,3))


    
    for i in range(n):
        summ = np.sum(1/nnyp[i])
        w = (1/nnyp)/summ
        en_p[i] = np.dot(w[i].T.reshape(1,3),my[i].reshape(3,1))

        if (en_p[i]>0.5):
            en_p[i]=1
        else:
            en_p[i]=0
    
    return en_p

def cmp(perdict, label):
    n, =label.shape
    wrong = []
    for i in range(n):
        if (perdict[i] != label[i]):
            wrong.append(i)
            
    return wrong 

def main(argv):
    data,X,y = manipulate_data()
    
    X_scale = scale_features(X, 0, 1)
    X_train, X_test, y_train, y_test = load_train_test_data(X_scale,y,train_ratio=.7)
    y_test.reshape(-1,1)
    y_train.reshape(-1,1)
    m,k = X_train.shape
    n,d = X_test.shape
    
    '''
    for i in range(n):
        if (y_test[i] ==1):
            plt.scatter(X_test[i,0],X_test[i,1], c = 'g')
        else:
            plt.scatter(X_test[i,0],X_test[i,1], c = 'r')
    '''
    
    kn,knp_tr,knp_te=cali_knn(X_train,y_train,X_test)
    kn = kn.reshape(n,1)
    kn_tr_l =np.zeros((m,1))
    knp_te_0 = np.delete(knp_te,1,1)
    knp_tr_0 = np.delete(knp_tr,1,1)
    knp_te_1 = np.delete(knp_te,0,1)
    knp_tr_1 = np.delete(knp_tr,0,1)
    
    
    
    log,logp_tr,logp_te=cali_logreg(X_train,y_train,X_test)
    log = log.reshape(n,1)
    log_tr_l =np.zeros((m,1))
    logp_te_0 = np.delete(logp_te,1,1)
    logp_tr_0 = np.delete(logp_tr,1,1)
    logp_te_1 = np.delete(logp_te,0,1)
    logp_tr_1 = np.delete(logp_tr,0,1)
    
    
    dec,decp_tr,decp_te=cali_dectree(X_train,y_train,X_test)
    dec = dec.reshape(n,1)
    dec_tr_l =np.zeros((m,1))
    decp_te_0 = np.delete(decp_te,1,1)
    decp_tr_0 = np.delete(decp_tr,1,1)
    decp_te_1 = np.delete(decp_te,0,1)
    decp_tr_1 = np.delete(decp_tr,0,1)
    
    
    
    '''
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, knp_te_1, n_bins=10)
    knbscore =  brier_score_loss(y_test, knp_te_1, pos_label=y.max())
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (%1.3f)" % ('clog', knbscore))
    ax2.hist(knp_tr_1, range=(0, 1), bins=10, label='clog',histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    '''
    
    
    for i in range(m):
        kn_tr_l[i] = abs(y_train[i] - knp_tr_1[i])
        log_tr_l[i] = abs(y_train[i] - logp_tr_1[i])
        dec_tr_l[i] = abs(y_train[i] - decp_tr_1[i])
        

    y_hat = np.hstack((np.hstack((kn,log)),dec))
    y_1_p_te =np.hstack((np.hstack((knp_te_1,logp_te_1)),decp_te_1))
    y_0_p_te =np.hstack((np.hstack((knp_te_0,logp_te_0)),decp_te_0))
    y_1_p_tr =np.hstack((np.hstack((knp_tr_1,logp_tr_1)),decp_tr_1))
    y_0_p_tr =np.hstack((np.hstack((knp_tr_0,logp_tr_0)),decp_tr_0))
    yp_te = np.hstack((y_1_p_te,y_0_p_te))
    yp_tr_l = np.hstack((np.hstack((kn_tr_l,log_tr_l)),dec_tr_l))
    print(yp_tr_l)
    nn_yp_te = nn(X_train,yp_tr_l,X_test)
    print(nn_yp_te)

    mvm = mv(kn,log,dec)
    #print(y)

    en_p = en(nn_yp_te,y_1_p_te)
    
    print("knn test : %f" % (sklearn.metrics.accuracy_score(kn, y_test)))
    print("log test : %f" % (sklearn.metrics.accuracy_score(log, y_test)))
    print("dec test : %f" % (sklearn.metrics.accuracy_score(dec, y_test)))
    #print("nn_p test R^2: %f" % (sklearn.metrics.r2_score(nn_yp_te, yp_te)))
    print("mv test : %f" % (sklearn.metrics.accuracy_score(mvm,y_test)))
    print("en_p test : %f" % (sklearn.metrics.accuracy_score(en_p,y_test)))
    knn_w = cmp(kn,y_test)
    log_w = cmp(log,y_test)
    dec_w = cmp(dec,y_test)
    en_p_w = cmp(en_p,y_test)
    '''print(knn_w)
    print(log_w)
    print(dec_w)
    print(en_p_w)'''
    
if __name__ == "__main__":
    main(sys.argv)