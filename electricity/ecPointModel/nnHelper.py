import os, random
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

def rho_risk2(y_pred,y_true,rho):
    assert len(y_pred) == len(y_true)
    diff1 = (y_true-y_pred)*rho*(y_true>=y_pred)
    diff2 = (y_pred-y_true)*(1-rho)*(y_true<y_pred)
    denominator = np.sum(y_true)
    return 2*(np.sum(diff1)+np.sum(diff2))/denominator

def rho_risk(y_pred,y_true,rho):
    assert len(y_pred) == len(y_true)
    diff = -np.sum(2*(y_pred-y_true)*(rho*(y_pred<=y_true)-(1-rho)*(y_pred>y_true)))
    denominator = np.sum(y_true)
    return diff/denominator
