import os, sys, math, random, time
from datetime import datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn

from nnModels import TCN
from nnTrainer import nn_trainer


### The input dataset
#ctx = mx.gpu(1)
ctx = mx.cpu(1)
with open('feature_prepare.pkl', 'rb') as f:
    [trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt] = pickle.load(f)


sub_train_X, sub_train_Y = nd.array(trainX_dt, ctx=ctx), nd.array(trainY_dt, ctx=ctx)
sub_valid_X, sub_valid_Y = nd.array(testX_dt, ctx=ctx), nd.array(testY_dt, ctx=ctx)
future_train_X, future_test_X = nd.array(trainY2_dt, ctx=ctx), nd.array(testY2_dt, ctx=ctx)

sub_train_nd = gluon.data.ArrayDataset(sub_train_X, future_train_X, sub_train_Y)

model1 = TCN()
#choose parameters
batch_size= 128
n_epochs=500
"""
The model training
"""
### The model parameters
abs_loss = gluon.loss.L1Loss()
L2_loss = gluon.loss.L2Loss()
huber_loss = gluon.loss.HuberLoss()
initializer = mx.initializer.MSRAPrelu()
optimizer = 'adam';
optimizer_params = {'learning_rate': 0.05}

trainer_params_list = {'batch_size': batch_size,'epoch_num':n_epochs,
                'loss_func': huber_loss, 'initializer': initializer,
                'optimizer':optimizer, 'optimizer_params':optimizer_params}
train_mark='testing'
nn_trainer(train_mark, model1, sub_train_nd, sub_valid_X,future_test_X, sub_valid_Y, trainer_params_list=trainer_params_list, ctx=ctx)
