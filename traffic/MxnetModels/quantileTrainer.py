import logging,math,os
from collections import deque
from tqdm import tqdm

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from utils import rhoRisk,rhoRisk2
from MxnetModels.quantileModels import QuantileLoss

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
        predResultQ10,predResultQ50 , predResultQ90  = nnModel(testX[0:batchSize,:], testX2[0:batchSize,:])
        for i in range(1, blockSize):
            subStartIndex = i*batchSize
            subEndIndex = min(subStartIndex+batchSize, predDtSize)
            blockPredQ10, blockPredQ50,blockPredQ90 = nnModel(testX[subStartIndex:subEndIndex,:], testX2[subStartIndex:subEndIndex,:])
            predResultQ10 = nd.concat(predResultQ10, blockPredQ10, dim=0)
            predResultQ50 = nd.concat(predResultQ50, blockPredQ50, dim=0)
            predResultQ90 = nd.concat(predResultQ90, blockPredQ90, dim=0)
        return predResultQ10, predResultQ50, predResultQ90
    
    def pointEvaluator(self, nnModel, testX, testX2, testY, lossFunc):
        pred = self.predict(nnModel, testX, testX2)
        validPred = pred.asnumpy()
        validTrue = testY
        print(validPred)
        print(validTrue)
        # The loss
        loss = nd.mean(lossFunc(pred, nd.array(testY, dtype='float32', ctx=self.dataCtx))).asscalar()
        # The evaluation metrics
        validND, validSMAPE, validNRMSE = ND(validPred, validTrue), SMAPE(validPred, validTrue), NRMSE(validPred, validTrue)
        return loss, validND,validSMAPE,validNRMSE
    
    def probEvaluator(self, nnModel, testX, testX2, testY, lossFunc):
        predQ10, predQ50, predQ90 = self.predict(nnModel, testX, testX2)
        validPredQ10,validPredQ50, validPredQ90 = predQ10.asnumpy(), predQ50.asnumpy(), predQ90.asnumpy()
        validTrue = testY
        # The loss
        loss = nd.sum(lossFunc(predQ50, nd.array(testY, dtype='float32', ctx=self.dataCtx))).asscalar()
        # The evaluation metrics
        rho50 = rhoRisk(validPredQ50.reshape(-1,), validTrue.reshape(-1,), 0.5)
        rho90 = rhoRisk(validPredQ90.reshape(-1,), validTrue.reshape(-1,),0.9)

        return loss, rho50, rho90


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
        ### The quantile loss
        loss10 = QuantileLoss(quantile_alpha=0.1)
        loss50= QuantileLoss(quantile_alpha=0.5)
        loss90 = QuantileLoss(quantile_alpha=0.9)
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
                    outputQ10, outputQ50, outputQ90 = self.model(data, convData)
                    lossQ10 = loss10(outputQ10, label)
                    lossQ50 = loss50(outputQ50, label)
                    lossQ90 = loss90(outputQ90,label)
                    loss = (lossQ10+lossQ50+lossQ90)#*weight

                loss.backward()
                trainer.step(batch_size=1,ignore_stale_grad=True)
                batchLoss = nd.sum(loss).asscalar()
                batchAvgLoss = batchLoss / data.shape[0]
                cumLoss += batchLoss
            #sampling
                cumSamples += batchSize
                if(cumSamples>maxTrainingSample): break
            #logging.info("Epoch %s / %s. Loss: %s." % (e + 1, epochs, cumLoss / nSamples))
            print("Epoch %s / %s. Training Loss: %s." % (e + 1, epochs, cumLoss / nSamples))
            lossTrainSeq.append(cumLoss/nSamples)
            if not testX is None:
                testLoss, rho50, rho90 = self.probEvaluator(self.model, testX, futureTestX, testY, lossFunc)
                print("Epoch %d, valid loss: %f valid rho-risk 50: %f,  valid rho-risk 90: %f" % (e+1, testLoss, rho50,rho90))
                ### The early stopping framework
                if(rho50 < bestValidMetric):
                    tmpModel = self.saveCheckpoint(self.model, 'Params', mark, rho50)
                    modelDeque.clear()
                    modelDeque.append(tmpModel)
                    ## update the best metrics
                    bestValidMetric = rho50
                elif (len(modelDeque)>0) and (len(modelDeque) < esEpochs):
                    modelDeque.append(tmpModel)
                elif (len(modelDeque)>0):
                    break
        bestModel = modelDeque.popleft()
        return history
