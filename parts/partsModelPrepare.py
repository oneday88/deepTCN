from tqdm import tqdm

import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime, timedelta
from sklearn import preprocessing

"""
Load the data
"""
parts = pd.read_csv("carparts.csv", index_col=0)
parts.fillna(0, inplace=True)

"""
Series selection:
(1) Removing series possessing fewer than ten positive monthly demands.
(2) Removing series having no positive demand in the first 15 and final 15 months
"""
cond1 = ((parts>0).sum(axis=1)>=10)
cond2 =  (parts.iloc[:,0:15].sum(axis=1)>0 ) & (parts.iloc[:,-15:].sum(axis=1)>0)
parts = parts.loc[cond1 & cond2,:].values

"""
The auxilary time index
"""
timeList = np.array(range(51))
yearList = (timeList /12).astype(int)
monthList = timeList % 12

partsIndex = list(range(1046))

"""
The loop for generation the model input
"""
movingWindowDis = 6;
sampleLen = 24; #12+12
inputLen = 12;
outputLen = 12;
totalN = 4; ## The total days
testN = 1    ## The testing days, day of the last 7 days
trainN = totalN - testN ## The training days

trainXList = [];trainYList = [];trainX2List = []
testXList = []; testYList = []; testX2List = []
for subParts in tqdm(partsIndex, desc='epochs'):
    print(subParts)
    subSeries = parts[subParts,:]

    trainX = np.zeros(shape=(trainN, inputLen))       ## The input series
    trainY = np.zeros(shape=(trainN, outputLen))      ## The output series

    testX = np.zeros(shape=(testN, inputLen))        ## The input series
    testY = np.zeros(shape=(testN, outputLen))        ## The output series

    covariateNum = 3  # other features : partsId,nYear,nMonth
    trainX2 = np.zeros(shape=(trainN, sampleLen, covariateNum))
    testX2 = np.zeros(shape=(testN, sampleLen, covariateNum))
    tsLen = subSeries.shape[0]
    startIndex = tsLen - sampleLen
    for i in range(totalN):
        ### The sequence data
        seriesX = subSeries[startIndex:startIndex+inputLen]    #168
        seriesY = subSeries[startIndex+inputLen:startIndex+sampleLen]  #24
        ### The covariate
        partsXY = np.repeat(subParts, sampleLen)
        nYearXY = yearList[startIndex:startIndex+sampleLen]
        nMonthXY = monthList[startIndex:startIndex+sampleLen]
        covariateXY = np.c_[partsXY,nYearXY,nMonthXY]  #192*6

        if(i<testN):
            testX[i] = seriesX
            testY[i] = seriesY
            testX2[i,:,:] = covariateXY

        else:
            trainX[i-testN] = seriesX
            trainY[i-testN] = seriesY
            trainX2[i-testN] = covariateXY
        # update the startIndex
        startIndex = startIndex - movingWindowDis

    testXList.append(testX)
    testX2List.append(testX2)
    testYList.append(testY)
    trainXList.append(trainX)
    trainX2List.append(trainX2)
    trainYList.append(trainY)


trainXDt = np.vstack(trainXList)
trainYDt = np.vstack(trainYList)
trainX2Dt = np.vstack(trainX2List)

testXDt = np.vstack(testXList)
testYDt = np.vstack(testYList)
testX2Dt = np.vstack(testX2List)

### Save the data
with open('partsPrepare.pkl', 'wb') as f:
    pickle.dump([trainXDt,trainX2Dt, trainYDt,testXDt, testX2Dt,testYDt], f, -1)
