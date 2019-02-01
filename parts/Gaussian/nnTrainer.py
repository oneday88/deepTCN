import sys, math, random, time, datetime
import _pickle as pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn,rnn

from tqdm import trange

from nnHelper import smape,rmsle,ND,NRMSE,rho_risk,rho_risk2
from nnModels import QuantileLoss


### check point: save the temporal model params
def save_checkpoint(net, mark, valid_metric, save_path):
    if not path.exists(save_path):
        os.makedirs(save_path)
    filename = path.join(save_path, "mark_{:s}_metrics_{:.3f}".format(mark, valid_metric))
    filename +='.param'
    net.save_params(filename)

def DLPred(net, dt):
    if(dt.shape[0]<=60000):
        print(type(net(conv_dt, dt)))
        return net(dt)
    block_size = dt.shape[0] //60000+1
    pred_result = net(dt[0:60000,])
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    return pred_result

def Gaussian_loss(mu, sigma, y):
    log_gaussian=-0.5* np.log(2* np.pi)- nd.log(sigma+0.01)- (y- mu)**2/(2*((sigma)**2)+0.0001)
    res=-nd.sum(log_gaussian, axis= 1)
    return nd.mean(res)

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
    #print('sss')
    return pred_result


"""
The main training process
"""
def nn_trainer(train_mark, model, train_data, test_conv_X, test_data_X,test_data_Y, trainer_params_list, ctx):
    """Parsing the params list"""
    ### The data
    batch_size = trainer_params_list['batch_size']
    epoches = trainer_params_list['epoch_num']

    loss_func = trainer_params_list['loss_func']
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
    ### The quantile loss
    ### The training process
    for epoch in trange(epoches):
        start=time.time()
        train_loss = 0
        k = 0
        train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
        for conv_data, data, label in train_iter:
            label = label.as_in_context(ctx)
            with autograd.record():
                pred_mu, pred_sigma = model(conv_data, data)
                loss = Gaussian_loss(pred_mu, pred_sigma, label)
            loss.backward()
            trainer.step(batch_size=1,ignore_stale_grad=True)
            train_loss += nd.sum(loss).asscalar()
            k += 1
            if k*batch_size>50000: 
                print('training_data_nb:',k*batch_size)
                break
        ### The test loss
        valid_true = test_data_Y.asnumpy()
        valid_mu, valid_sigma = DLPred2(model,test_conv_X, test_data_X)  
        
        valid_loss = nd.sum(loss_func(nd.array(valid_true), nd.array(valid_mu))).asscalar()     
        rho50 = rho_risk(valid_mu.asnumpy(), test_data_Y.asnumpy(), 0.5)        
        valid_pred90 = norm.ppf(0.9,valid_mu.asnumpy(), valid_sigma.asnumpy())
        rho90 = rho_risk(valid_pred90, test_data_Y.asnumpy(), 0.9)

        ### The evaluation
        #rho90_10 = rho_risk(0,31, valid_pred90_10.asnumpy(), valid_true_10.asnumpy(), 0.9)
        print("Epoch %d, valid loss: %f valid rho-risk 50: %f,  valid rho-risk 90: %f" % (epoch, valid_loss, rho50,rho90))
        
