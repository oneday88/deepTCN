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
def DLPreprocess(dt, catFeatureList=None, numericFeatureList=None):
    # label encoding of the categorical features
    labelEncList = []
    if catFeatureList is not None:
        for categoryFeature in catFeatureList:
            labelEnc = preprocessing.LabelEncoder()
            labelEnc.fit(dt.loc[:, categoryFeature])
            labelEncList.append(labelEnc)
            dt.loc[:, categoryFeature] = labelEnc.transform(dt.loc[:, categoryFeature])
    # numeric feature normalization
    if numericFeatureList is not None:
        dt[numericFeatureList] = preprocessing.scale(dt[numericFeatureList])
    return dt, labelEncList

"""
Function for evaluation
"""
def SMAPE(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    denominator = (np.abs(yTrue) + np.abs(yPred))
    diff = np.abs(yTrue - yPred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)

def ND(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    demoninator = np.sum(np.abs(yTrue))
    diff = np.sum(np.abs(yTrue - yPred))
    return 1.0*diff/demoninator

def RMSLE(yPred, yTrue) :
    assert len(yPred) == len(yTrue)
    assert len(yTrue) == len(yPred)
    return np.sqrt(np.mean((np.log(1+yPred) - np.log(1+yTrue))**2))

def NRMSE(yPred, yTrue):
    assert len(yPred) == len(yTrue)
    denominator = np.mean(yTrue)
    diff = np.sqrt(np.mean(((yPred-yTrue)**2)))
    return diff/denominator

def rhoRisk2(yPred,yTrue,rho):
    assert len(yPred) == len(yTrue)
    diff1 = (yTrue-yPred)*rho*(yTrue>=yPred)
    diff2 = (yPred-yTrue)*(1-rho)*(yTrue<yPred)
    denominator = np.sum(yTrue)
    return 2*(np.sum(diff1)+np.sum(diff2))/denominator

def rhoRisk(yPred,yTrue,rho):
    assert len(yPred) == len(yTrue)
    diff = -np.sum((yPred-yTrue)*(rho*(yPred<=yTrue)-(1-rho)*(yPred>yTrue)))
    denominator = np.sum(yTrue)
    return diff/denominator
