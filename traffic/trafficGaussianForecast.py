import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd

"""
Load the model data
"""
with open('trafficPrepare.pkl', 'rb') as f:
    [trainXDt,trainX2Dt, trainYDt,testXDt, testX2Dt,testYDt] = pickle.load(f)

# cpu or gpu
modelCtx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
dataCtx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

"""
The model training
"""
from MxnetModels.gaussianModels import TCN
from MxnetModels.gaussianTrainer import nnTrainer
# The models
inputSize=168
outputSize=24
dilations = [1,2,4,8,16,32]
nResidue = 35
actType='relu'
dropout=0.2
model1 = TCN(inputSize, outputSize, dilations,nResidue, actType, dropout)

mlpTrainer = nnTrainer(model1, dataCtx, modelCtx)

"""
define the trainer
"""
from mxnet.gluon.loss import L2Loss, L1Loss, HuberLoss

epochs = 100
esEpochs = 10 #
evalCriteria = 'min'

batchSize = 64
learningRate = 0.001
sampleRate = 0.3

initializer = mx.init.Xavier(magnitude=2.24)
initializer = mx.initializer.MSRAPrelu()
optimizer = 'adam';
lossFunc = HuberLoss() #Which is robust to outlier

paramsList = {'epochs': epochs, 'esEpochs': esEpochs, 'evalCriteria': evalCriteria,
        'batchSize': batchSize, 'learningRate': learningRate, 'sampleRate': sampleRate, 
                    'initializer': initializer, 'optimizer':optimizer, 'lossFunc': lossFunc}

##The model traning
trainingMark='Normal'
trainHistory = mlpTrainer.fit(trainingMark,trainXDt,trainX2Dt, trainYDt, testXDt, testX2Dt,testYDt, paramsList)
