import sys, math, random, time, datetime
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import autograd, gluon, nd, gpu
from mxnet.gluon import nn,rnn

from nnHelper import smape,rmsle,ND,NRMSE,rho_risk,all_avg,avg_rho_risk
import time

### check point: save the temporal model params
def save_checkpoint(net, mark, valid_metric, save_path):
    if not path.exists(save_path):
        os.makedirs(save_path)
    filename = path.join(save_path, "mark_{:s}_metrics_{:.3f}".format(mark, valid_metric))
    filename +='.param'
    net.save_params(filename)

def DLPred(net, dt):
    if(dt.shape[0]<=60000):
        return net(dt)[0], net(dt)[1]
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
    #print('comein')
    #print(type(conv_dt),type(dt))
    if(dt.shape[0]<=60000):
        #print(type(net(conv_dt, dt)))
        return net(conv_dt, dt)[0], net(conv_dt, dt)[1]
    block_size = dt.shape[0] //60000+1
    pred_result = net(conv_dt[0:60000,], dt[0:60000,])
    #print('comein')
    for i in range(1,block_size):
        i = i*60000
        j = min(i+60000, dt.shape[0])
        block_pred = net(conv_dt[i:j, ], dt[i:j, ])
        pred_result = nd.concat(pred_result, block_pred, dim=0)
    #print(type(pred_result))
    #print('why')
    return pred_result

## obtain test sets for Oct and Nov respectively
def delete_rows(a):
    b = np.count_nonzero(a,axis=1)
    return  b==(a.shape[1])
    
    





"""
The main training process
"""
def nn_trainer(train_mark, model, train_data, test_conv_X, test_data_X,test_data_Y, trainer_params_list, ctx):
    """Parsing the params list"""
    start=time.time()
    ### The data
    batch_size = trainer_params_list['batch_size']
    epochs = trainer_params_list['epoch_num']

    loss50 = trainer_params_list['loss50']
    loss90 = trainer_params_list['loss90']
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
    best_loss_10 = 1000000000000
    best_loss_11 = 1000000000000
    for e in range(epochs):
        start1=time.time()
        train_loss = 0
        k = 0
        train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
        for conv_data, data, label in train_iter:
            label = label.as_in_context(ctx)
            with autograd.record():
                output_q50, output_q90 = model( data, conv_data)
                #print(type(output_q50))
                loss_q50 = loss50(output_q50, label)
                loss_q90 = loss90(output_q90, label)
                loss = loss_q50 + loss_q90 
     
            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)
            train_loss += nd.sum(loss).asscalar()
            #print(train_loss)
            k += 1
            if k*batch_size>n_train*1.2: 
                print('training_data_nb:',k*batch_size)
                break
        ### The test loss

        valid_pred50, valid_pred90 = DLPred2(model,  test_data_X,test_conv_X)
        

        valid_q50 = loss50(valid_pred50,test_data_Y)
        
        valid_q90 = loss90(valid_pred90,test_data_Y)
        
        valid_loss = (nd.sum(valid_q50+valid_q90).asscalar())
        


        print("Epoch %d, valid_loss: %f" % (e, (valid_loss)))

       ## rho_risk

        #rho50 = rho_risk( valid_pred50.asnumpy(), test_data_Y.asnumpy(), 0.5)
        #rho90 = rho_risk( valid_pred90.asnumpy(), test_data_Y.asnumpy(), 0.9)
        avg_rho50 = avg_rho_risk( valid_pred50.asnumpy(), test_data_Y.asnumpy(), 0.5,7)
        avg_rho90 = avg_rho_risk( valid_pred90.asnumpy(), test_data_Y.asnumpy(), 0.9,7)
    
        #print("Epoch %d, rho50: %f,rho90: %f" % (e, rho50,rho90))
        print("Epoch %d, avg_rho50: %f,avg_rho90: %f" % (e, avg_rho50,avg_rho90))
        
        end1=time.time()  
        print('time for one epoch:', str(round((end1-start1),2))+'s')

    end=time.time()
    print('total_time:', end-start)
    
