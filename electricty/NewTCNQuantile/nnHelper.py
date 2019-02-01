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

def group_ND(y_pred, y_true, series_num):#The dimension of the y_pred 2590*24, the dimension of the y_true 2590*24"""
    assert y_pred.shape == y_true.shape
    group0 = np.array(range(series_num))*7
    group1 = group0+1
    group2 = group0+2
    group3 = group0+3
    group4 = group0+4
    group5 = group0+5
    group6 = group0+6
    ND0 = ND(y_pred[group0],y_true[group0])
    ND1 = ND(y_pred[group1],y_true[group1])
    ND2 = ND(y_pred[group2],y_true[group2])
    ND3 = ND(y_pred[group3],y_true[group3])
    ND4 = ND(y_pred[group4],y_true[group4])
    ND5 = ND(y_pred[group5],y_true[group5])
    ND6 = ND(y_pred[group6],y_true[group6])
    meanND = np.mean([ND0,ND1,ND2,ND3,ND4,ND5,ND6])
    return meanND,ND0,ND1,ND2,ND3,ND4,ND5,ND6
    
def group_NRMSE(y_pred, y_true, series_num):#The dimension of the y_pred 2590*24, the dimension of the y_true 2590*24"""
    assert y_pred.shape == y_true.shape
    group0 = np.array(range(series_num))*7
    group1 = group0+1
    group2 = group0+2
    group3 = group0+3
    group4 = group0+4
    group5 = group0+5
    group6 = group0+6
    NRMSE0 = NRMSE(y_pred[group0],y_true[group0])
    NRMSE1 = NRMSE(y_pred[group1],y_true[group1])
    NRMSE2 = NRMSE(y_pred[group2],y_true[group2])
    NRMSE3 = NRMSE(y_pred[group3],y_true[group3])
    NRMSE4 = NRMSE(y_pred[group4],y_true[group4])
    NRMSE5 = NRMSE(y_pred[group5],y_true[group5])
    NRMSE6 = NRMSE(y_pred[group6],y_true[group6])
    meanNRMSE = np.mean([NRMSE0,NRMSE1,NRMSE2,NRMSE3,NRMSE4,NRMSE5,NRMSE6])
    return meanNRMSE,NRMSE0,NRMSE1,NRMSE2,NRMSE3,NRMSE4,NRMSE5,NRMSE6  

def group_rho_risk(y_pred, y_true, series_num,rho):#The dimension of the y_pred 2590*24, the dimension of the y_true 2590*24"""
    assert y_pred.shape == y_true.shape
    group0 = np.array(range(series_num))*7
    group1 = group0+1
    group2 = group0+2
    group3 = group0+3
    group4 = group0+4
    group5 = group0+5
    group6 = group0+6
    risk0 = rho_risk(y_pred[group0],y_true[group0], rho)
    risk1 = rho_risk(y_pred[group1],y_true[group1], rho)
    risk2 = rho_risk(y_pred[group2],y_true[group2], rho)
    risk3 = rho_risk(y_pred[group3],y_true[group3], rho)
    risk4 = rho_risk(y_pred[group4],y_true[group4], rho)
    risk5 = rho_risk(y_pred[group5],y_true[group5], rho)
    risk6 = rho_risk(y_pred[group6],y_true[group6], rho)
    meanrisk = np.mean([risk0,risk1,risk2,risk3,risk4,risk5,risk6])
    return meanrisk,risk0,risk1,risk2,risk3,risk4,risk5,risk6  


    

