import os, math,argparse
import _pickle as pickle
from datetime import datetime, timedelta
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
Load the data and preprocessing
"""
# load the data
#ecPath = '/Users/yitianchen/Work/ProbabilisticForecasting/VAE/electricity'
ecPath = '/root/Oneday/deeptcn/ec'
dt = np.loadtxt(os.path.join(ecPath,"modelData.csv"), dtype=np.str, delimiter=",")

# The date-hour index for the ec data
dateList = pd.date_range(start='01/01/2012', end='31/12/2014')
dateList = pd.to_datetime(dateList)

hourList = []
for nDate in dateList:
    for nHour in range(24):
        tmpTimestamp = nDate+timedelta(hours=nHour)
        hourList.append(tmpTimestamp)
hourList = np.array(hourList)

# The station List 0:370
#stationIndex = list(range(370))
stationIndex = list(range(100))
"""
Prepare data for VAEs model training
"""
slidingWindowDis=24;
# input seq size: 168; output seq size: 24
sampleLen = 192;    inputLen = 168; outputLen = 24;

# total number of days of the series
totalN = 500;               # The total days
testN = 7                   # The last 7 days is used for evaluation
trainN = totalN - testN     # The training days

trainXList = [];trainYList = [];trainX2List = []
testXList = []; testYList = []; testX2List = []
for station in tqdm(stationIndex, desc='epochs'):
    print(station)
    subSeries = dt[station,:].astype('float32')
    subSeries = np.log(subSeries+1)
    # The input seq and output sequence of training dataset
    trainX = np.zeros(shape=(trainN, inputLen))
    trainY = np.zeros(shape=(trainN, outputLen))
    
    # The input seq and output sequence of testing dataset
    testX  = np.zeros(shape=(testN, inputLen))
    testY = np.zeros(shape=(testN, outputLen))

    # The auxiliary covariate features for prediction
    covarNum = 6   # [station_id, nYear, nMonth, day_of_month, day_of_week, iHour]
    trainX2 = np.zeros(shape=(trainN, sampleLen, covarNum))
    testX2 = np.zeros(shape=(testN, sampleLen, covarNum))
    

    # the length of the series 
    tsLen = len(subSeries)
    startIndex = tsLen-sampleLen
    for i in range(totalN):
        # The encoder and decoder series data
        seriesX = subSeries[startIndex:startIndex+inputLen]
        seriesY = subSeries[startIndex+inputLen:startIndex+sampleLen]
        if(sum(seriesX==0)>=72):break
        """
        The auxiliary variables to improve the prediction accuracy
        """
        ### The covariate
        stationXY = np.repeat(station, sampleLen)
        # the timestamp index
        ### the time index
        timeIndexXY = pd.to_datetime(hourList[startIndex:startIndex+sampleLen])

        nYearXY = timeIndexXY.year-2012
        nMonthXY = timeIndexXY.month-1
        mDayXY = timeIndexXY.dayofyear-1
        wDayXY = timeIndexXY.weekday
        nHourXY = timeIndexXY.hour
    
        covariateXY = np.c_[stationXY,nYearXY,nMonthXY,mDayXY,wDayXY,nHourXY]
        """
        The tensor data of training dataset and testing dataset
        """
        if(i<testN):
            testIndex = i
            testX[testIndex] = seriesX
            testY[testIndex] = seriesY
            testX2[testIndex] = covariateXY
        else:

            trainX[i-testN] = seriesX
            trainY[i-testN] = seriesY
            trainX2[i-testN] = covariateXY
        # create the training input-output sequence by sliding window approach
        startIndex = startIndex - slidingWindowDis
  

    testXList.append(testX)
    testX2List.append(testX2)
    testYList.append(testY)
    
    trainXList.append(trainX)
    trainX2List.append(trainX2)
    trainYList.append(trainY)

testXDt = np.vstack(testXList)
testYDt = np.vstack(testYList)
testX2Dt = np.vstack(testX2List)

trainXDt = np.vstack(trainXList)
trainYDt = np.vstack(trainYList)
trainX2Dt = np.vstack(trainX2List)

##Do normalization to the original series
scaler = preprocessing.StandardScaler()
scaler.fit(trainXDt)

trainXDt = scaler.transform(trainXDt)
testXDt = scaler.transform(testXDt)

"""
### Select the data of the Dec
isDec = trainX2Dt[:,:,2] == 11
trainXDt = trainXDt[isDec[:,0]]
trainYDt = trainYDt[isDec[:,0]]
trainX2Dt = trainX2Dt[isDec[:,0]]
"""

### Save the data
fileName = 'ecLogPrepare.pkl'
with open(fileName, 'wb') as f:
    pickle.dump([trainXDt,trainX2Dt, trainYDt,testXDt, testX2Dt,testYDt], f, -1)
