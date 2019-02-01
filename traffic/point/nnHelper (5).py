import os
import random
from os import path
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, gpu, autograd
from mxnet.gluon import nn, rnn
from sklearn import preprocessing

"""
Function for data preprocess
"""
def DLPreprocess(dt, cat_feature_list, numeric_feature_list):
    ### label encode of categorical features
    label_enc_list = []
    for category_feature in cat_feature_list:
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(dt.loc[:, category_feature])
        label_enc_list.append(label_enc)
        dt.loc[:, category_feature] = label_enc.transform(dt.loc[:, category_feature])
    ### numeric feature normalization
    dt[numeric_feature_list] = preprocessing.scale(dt[numeric_feature_list])
    return dt,label_enc_list

"""
Function for evaluation
"""
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def ND(y_pred, y_true):
    demoninator = np.sum(np.abs(y_true))
    diff = np.sum(np.abs(y_true - y_pred))
    return 1.0*diff/demoninator

def rmsle(y_pred, y_true) :
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_true))**2))

def NRMSE(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    denominator = np.mean(y_true)
    diff = np.sqrt(np.mean(((y_pred-y_true)**2)))
    return diff/denominator

def MAE(y_pred, y_true):
    assert len(y_true) == len(y_pred)
    assert np.all(y_true)>=0
    assert np.all(y_pred)>=0
    gap = np.abs(y_pred-y_true)
    return np.mean(gap)

def NMAE(y_pred, y_true, weight, m):
    assert len(y_true) == len(y_pred)
    assert np.all(y_true)>=0
    assert np.all(y_pred)>=0
    gap = np.abs(y_pred-y_true)
    result = np.mean(gap*weight/m)
    return result


def Gaussian_loss(mu, sigma, y):
    log_gaussian=-0.5* np.log(2* np.pi)- nd.log(sigma+0.01)- (y- mu)**2/(2*((sigma)**2)+0.0001)
    res=-nd.sum(log_gaussian, axis= 1)
    return nd.mean(res)



def rho_risk(pre,tar,rho):
    #pre = np.sum(pred[:,L:(L+S)],axis=1)
    #tar = np.sum(target[:,L:(L+S)],axis=1)
    diff = -np.sum(2*(pre-tar)*(rho*(pre<=tar)-(1-rho)*(pre>tar)))
    denominator = np.sum(tar)
    return diff/denominator





def all_avg(K, pred, target, rho):
    rho_all = 0
    for i in range(K-1):
        rho_all+=rho_risk(i,1,pred,target,rho)
    return np.mean(rho_all)
