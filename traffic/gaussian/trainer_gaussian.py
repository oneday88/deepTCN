import sys, math, random, time, datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn,rnn

from nnHelper import smape,rmsle,ND,NRMSE,rho_risk
import time
from scipy.stats import norm
from nnHelper import Gaussian_loss, avg_rho_risk

### check point: save the temporal model params
def save_checkpoint(net, mark, valid_metric, save_path):
    if not path.exists(save_path):
        os.makedirs(save_path)
    filename = path.join(save_path, "mark_{:s}_metrics_{:.3f}".format(mark, valid_metric))
    filename +='.param'
    net.save_params(filename)

def DLPred(net, dt):
    if(dt.shape[0]<=60000):
        return net(dt)
    block_size = dt.shape[0] //60000+1
    pred_result = net(dt[0:60000,])
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    return pred_result


"""
The main training process
"""
def DLPred2(net,conv_dt ,dt):
    if(dt.shape[0]<=60000):
        return net(conv_dt, dt)
    block_size = dt.shape[0] //60000+1
    pred_result = net(conv_dt[0:60000,], dt[0:60000,])
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(conv_dt[i:j, ], dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    return pred_result


"""
The main training process
"""
def nn_trainer(train_mark, model, train_data, test_conv_X, test_data_X,test_data_Y, trainer_params_list, ctx):
    """Parsing the params list"""
    ### The data
    batch_size = trainer_params_list['batch_size']
    epochs = trainer_params_list['epoch_num']

    loss_func = Gaussian_loss
    initializer = trainer_params_list['initializer']
    optimizer = trainer_params_list['optimizer']
    optimizer_params = trainer_params_list['optimizer_params']

    #train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    ### The model
    mx.random.seed(123456)
    model.collect_params().initialize(initializer, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),optimizer=optimizer, optimizer_params=optimizer_params)
    n_train = len(train_data)
    n_test = len(test_data_Y)
    ### The training process
    for e in range(epochs):
        start=time.time()
        train_loss = 0
        k = 0
        train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
        for conv_data, data, label in train_iter:
            label = label.as_in_context(ctx)
            with autograd.record():
                output_mu, output_sigma = model( data, conv_data)
                loss = loss_func(output_mu, output_sigma, label)
            loss.backward()
            trainer.step(1,ignore_stale_grad=True)
            train_loss += nd.sum(loss).asscalar()
            k += 1
            if k*batch_size>n_train*0.3: 
                print('training_data_nb:',k*batch_size)
                break
        ### The test loss
        valid_mu, valid_sigma  = DLPred2(model,  test_data_X,test_conv_X)
        
        valid_loss = loss_func(valid_mu, (valid_sigma),(test_data_Y)).asscalar()
        
        #rho50 = rho_risk(0,7, valid_mu.asnumpy(), test_data_Y.asnumpy(), 0.5)
        avg_rho50 = avg_rho_risk(valid_mu.asnumpy(), test_data_Y.asnumpy(), 0.5, 7)
        
        valid_pred90 = norm.ppf(0.9, valid_mu.asnumpy(), valid_sigma.asnumpy())
      #  print(valid_mu[0:5,:])
      #  print( valid_sigma[0:5,:])
      #  print(valid_pred90[0:5,:])
        
       # rho90 = rho_risk(0,7, valid_pred90, test_data_Y.asnumpy(), 0.9)
        avg_rho90 = avg_rho_risk(valid_pred90, test_data_Y.asnumpy(), 0.9,7)
       
        #print("Epoch %d, valid loss: %f rho50: %f, rho90 %f" % (e, valid_loss, rho50,rho90))
        print("Epoch %d, valid loss: %f avg_rho50: %f, avg_rho90 %f" % (e, valid_loss, avg_rho50,avg_rho90))
        end=time.time()
        print('total_time:', end-start)
