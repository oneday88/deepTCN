from tqdm import tqdm

import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime, timedelta
from sklearn import preprocessing

"""
Load the data
"""
traffic = pd.read_csv("traffic.csv", header=None,index_col=0).values.T
# remove the holiday based on the description of the data set 
dateList = pd.date_range(start = '01/01/2008', end='30/03/2009')
dateRemove  = pd.to_datetime(['2008-01-01', '2008-01-21', '2008-02-18','2008-03-31','2008-03-09', '2008-05-26','2008-07-04','2008-09-01','2008-11-11','2008-11-17', '2008-12-25', '2009-01-01','2009-01-21','2009-02-16',"2009-03-08"])
dateList = [s for s in dateList if s not in dateRemove]

hourList = []
for nDate in dateList:
    for nHour in range(24):
        tmpTimestamp = nDate+timedelta(hours=nHour)
        hourList.append(tmpTimestamp)
hourList = np.array(hourList)

stationIndex = list(range(963))

movingWindowDis = 24;
sampleLen = 192; #168+24
inputLen = 168;
outputLen = 24;
totalN = 440-7; ## The total days
testN = 7     ## The testing days, day of the last 7 days
trainN = totalN - testN ## The training days

trainXList = [];trainYList = [];trainX2List = []
testXList = []; testYList = []; testX2List = []
for station in tqdm(stationIndex, desc='epochs'):
    print(station)
    subSeries = traffic[station,:]

    trainX = np.zeros(shape=(trainN, inputLen))       ## The input series
    trainY = np.zeros(shape=(trainN, outputLen))      ## The output series

    testX = np.zeros(shape=(testN, inputLen))        ## The input series
    testY = np.zeros(shape=(testN, outputLen))        ## The output series

    covariateNum = 6   # other features covariateNum: stationId,nYear,nMonth,day_of_month, day_of_week, iHour
    trainX2 = np.zeros(shape=(trainN, sampleLen, covariateNum))
    testX2 = np.zeros(shape=(testN, sampleLen, covariateNum))
    ### Testing samples (7+1)*24
    tsLen = subSeries.shape[0]
    startIndex = tsLen - sampleLen
    for i in range(totalN):
        ### The sequence data
        seriesX = subSeries[startIndex:startIndex+inputLen]    #168
        seriesY = subSeries[startIndex+inputLen:startIndex+sampleLen]  #24
        ### The covariate
        stationXY = np.repeat(station, sampleLen)
        ### the time index
        timeIndexXY = pd.to_datetime(hourList[startIndex:startIndex+sampleLen])
        #print(timeIndexXY)
        nYearXY = timeIndexXY.year-2008
        nMonthXY = timeIndexXY.month-1
        mDayXY = timeIndexXY.day-1
        wDayXY = timeIndexXY.weekday
        nHourXY = timeIndexXY.hour

        covariateXY = np.c_[stationXY,nYearXY,nMonthXY,mDayXY,wDayXY,nHourXY]  #192*6

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
with open('trafficPrepare.pkl', 'wb') as f:
    pickle.dump([trainXDt,trainX2Dt, trainYDt,testXDt, testX2Dt,testYDt], f, -1)
