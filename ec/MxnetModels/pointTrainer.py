import logging,math,os
from collections import deque
from tqdm import tqdm

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from utils import SMAPE,RMSLE,ND,NRMSE

class nnTrainer(object):
    def __init__(self, nnModel, modelCtx, dataCtx):
        self.modelCtx = modelCtx
        self.dataCtx = dataCtx
        self.model =  nnModel

    def saveCheckpoint(self,nnModel, savePath, mark, testAuc):
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        filename = os.path.join(savePath, "mark_{:s}_metrics_{:.3f}".format(mark, testAuc))
        filename +='.param'
        nnModel.save_parameters(filename)
        return filename

    def predict(self,nnModel, testX,testX2, batchSize=3000):
        predDtSize = testX.shape[0]
        testX = nd.array(testX, dtype='float32', ctx=self.dataCtx)
        testX2 = nd.array(testX2, dtype='float32', ctx=self.dataCtx)
        # if the test dataset is small
        if(predDtSize<=batchSize):
            return nnModel(testX, testX2)
        # if the test dataset is large(to prevent memory allocation error of mxnet)
        blockSize = math.ceil(predDtSize // batchSize)+1
        predResult  = nnModel(testX[0:batchSize,:], testX2[0:batchSize,:])
        for i in range(1, blockSize):
            subStartIndex = i*batchSize
            subEndIndex = min(subStartIndex+batchSize, predDtSize)
            blockPred = nnModel(testX[subStartIndex:subEndIndex,:], testX2[subStartIndex:subEndIndex,:])
            predResult = nd.concat(predResult, blockPred, dim=0)
        return predResult
    
    def pointEvaluator(self, nnModel, testX, testX2, testY, lossFunc, mode='Normal'):
        assert mode in set(['Normal','logTransform'])
        pred = self.predict(nnModel, testX, testX2)
        validPred = pred.asnumpy()
        validTrue = testY
        if(mode =='logTransform'):
            validPred = np.exp(validPred)-1
            validTrue = np.exp(validTrue)-1
        # The loss
        loss = nd.mean(lossFunc(pred, nd.array(testY, dtype='float32', ctx=self.dataCtx))).asscalar()
        # The evaluation metrics
        validND, validSMAPE, validNRMSE = ND(validPred, validTrue), SMAPE(validPred, validTrue), NRMSE(validPred, validTrue)
        return loss, validND,validSMAPE,validNRMSE

    def fit(self, mark,  trainX, futureTrainX, trainY, testX, futureTestX, testY, paramsDict):
        """
        The parameters list:
            esEpochs: the early-stopping epoch: -1, non early stopping
        """
        epochs = paramsDict['epochs']
        esEpochs = paramsDict['esEpochs']
        evalCriteria = paramsDict['evalCriteria']

        batchSize = paramsDict['batchSize']
        learningRate = paramsDict['learningRate']
        sampleRate = paramsDict['sampleRate']

        lossFunc = paramsDict['lossFunc']
        optimizer = paramsDict['optimizer']
        initializer = paramsDict['initializer']
        ### The model initialization
        self.model.collect_params().initialize(initializer, ctx=self.modelCtx)
        ### The trainer
        trainer = gluon.Trainer(self.model.collect_params(), optimizer=optimizer, optimizer_params={'learning_rate': learningRate})

        ## 
        nSamples = trainX.shape[0]
        nBatch = int(nSamples / batchSize)
        maxTrainingSample = nSamples*sampleRate

        # Keep the metrics history
        history = dict()
        lossTrainSeq = []
        lossTestSeq = []

        # The early stopping framework
        bestValidMetric = 999999. if evalCriteria =='min' else 0
        if(esEpochs > 0):
            modelDeque= deque()

        for e in tqdm(range(epochs), desc='epochs'):
            cumLoss = 0.
            cumSamples = 0.
            trainIter = gluon.data.DataLoader(gluon.data.ArrayDataset(trainX, futureTrainX, trainY), batch_size=batchSize, shuffle=True)
            for  data, convData, label in trainIter:
                data = nd.array(data, dtype='float32', ctx=self.dataCtx)
                convData = nd.array(convData, dtype='float32', ctx=self.dataCtx)
                label = nd.array(label, dtype='float32', ctx=self.dataCtx)
                with autograd.record():
                    pred  = self.model(data, convData)
                    loss = lossFunc(pred, label)+lossFunc(nd.mean(pred,axis=1), nd.mean(label,axis=1))*0.2
                loss.backward()
                trainer.step(batch_size=data.shape[0], ignore_stale_grad=True)
                batchLoss = nd.sum(loss).asscalar()
                batchAvgLoss = batchLoss / data.shape[0]
                cumLoss += batchLoss
            #sampling
                cumSamples += batchSize
                if(cumSamples>maxTrainingSample): break
            #logging.info("Epoch %s / %s. Loss: %s." % (e + 1, epochs, cumLoss / nSamples))
            print("Epoch %s / %s. Training Loss: %s." % (e + 1, epochs, cumLoss / nSamples))
            logging.info("Epoch %s / %s. Training Loss: %s." % (e + 1, epochs, cumLoss / nSamples))
            lossTrainSeq.append(cumLoss/nSamples)
            if not testX is None:
                testLoss, testND,testSMAPE, testNRMSE = self.pointEvaluator(self.model, testX, futureTestX, testY, lossFunc, mark)
                print("Epoch %s / %s. Testing loss: %s. Testing ND: %s. Test SMAPE: %s. Test NRMSE: %s" % (e + 1, epochs, testLoss, testND, testSMAPE, testNRMSE))
                logging.info("Epoch %s / %s. Testing loss: %s. Testing ND: %s. Test SMAPE: %s. Test NRMSE: %s" % (e + 1, epochs, testLoss, testND, testSMAPE, testNRMSE))
                ### The early stopping framework
                if(testLoss < bestValidMetric):
                    tmpModel = self.saveCheckpoint(self.model, 'Params', mark, testLoss)
                    modelDeque.clear()
                    modelDeque.append(tmpModel)
                    ## update the best metrics
                    bestValidMetric = testLoss
                elif (len(modelDeque)>0) and (len(modelDeque) < esEpochs):
                    modelDeque.append(tmpModel)
                elif (len(modelDeque)>0):
                    break
        bestModel = modelDeque.popleft()
        return history
