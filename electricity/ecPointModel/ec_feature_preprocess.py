import os, math
import _pickle as pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn import preprocessing

### load the data

dir_path = '/data2/yitian/kdd2019deepTCN/electricity/TCNQuantileCheck'
dt = np.loadtxt(os.path.join(dir_path,"modelData.csv"), dtype=np.str, delimiter=",")

## The date range
date_list = pd.date_range(start='01/01/2012', end='31/12/2014')
date_list = pd.to_datetime(date_list)

hour_list = []
for nDate in date_list:
    for nHour in range(24):
        tmp_timestamp = nDate+timedelta(hours=nHour)
        hour_list.append(tmp_timestamp)
hour_list = np.array(hour_list)


station_index = list(range(370))

sliding_window_dis = 24;
sample_len = 192; #168+24
input_len = 168;
output_len = 24;
total_n = 800; ## The total days
test_n = 7     ## The testing days, day of the last 7 days
train_n = total_n - test_n ## The training days

trainX_list = [];trainX2_list = [];trainY_list = [];trainY2_list = []
testX_list = [];testX2_list = [];testY_list = [];testY2_list = []
#for station in station_index:
for station in station_index:
    print(station)
    sub_series = dt[station,1:].astype('float32')
    sub_index = np.array(range(26304))-np.min(np.where(sub_series>0))
    trainX = np.zeros(shape=(train_n, input_len))       ## The input series
    trainY = np.zeros(shape=(train_n, output_len))      ## The output series  

    testX  = np.zeros(shape=(7, input_len))        ## The input series
    testY = np.zeros(shape=(7, output_len))        ## The output series

    covariate_num = 8   # other features covariate_num: sub_index, station_id,nYear,nMonth,day_of_month, day_of_week, iHour
    trainX2 = np.zeros(shape=(train_n, input_len,covariate_num))
    trainY2 = np.zeros(shape=(train_n, output_len,covariate_num))
    testX2 = np.zeros(shape=(7, input_len,covariate_num))
    testY2 = np.zeros(shape=(7, output_len,covariate_num))
    ### Testing samples (7+1)*24
    ts_len = sub_series.shape[0]
    start_index = ts_len-sample_len
    for i in range(total_n):
        ### The sequence data
        series_x = sub_series[start_index:start_index+input_len]
        series_y = sub_series[start_index+input_len:start_index+sample_len]
        ### The index data
        hour_mean = np.mean(series_x.reshape(-1,24),axis=0)
        index_x = np.tile(hour_mean,7)
        index_y = np.tile(hour_mean,1)
        ### The covariate
        station_X = np.repeat(station, input_len)
        station_Y = np.repeat(station, output_len)
        ### the time index
        time_index_x = pd.to_datetime(hour_list[start_index:start_index+input_len])
        time_index_y = pd.to_datetime(hour_list[start_index+input_len:start_index+sample_len])
        nYear_X, nYear_Y = time_index_x.year-2012, time_index_y.year-2012
        nMonth_X, nMonth_Y = time_index_x.month-1, time_index_y.month-1
        mDay_X, mDay_Y = time_index_x.day-1, time_index_y.day-1
        wDay_X, wDay_Y = time_index_x.weekday, time_index_y.weekday
        nHour_X, nHour_Y = time_index_x.hour, time_index_y.hour
        holiday_X,holiday_Y = (mDay_X==24),(mDay_Y==24)
    
        covariate_X = np.c_[station_X,index_x,nYear_X,nMonth_X,mDay_X,wDay_X,nHour_X,holiday_X]
        covariate_Y = np.c_[station_Y,index_y,nYear_Y,nMonth_Y,mDay_Y,wDay_Y,nHour_Y,holiday_Y]
    
        if(i<test_n):
            test_index = i
            testX[test_index] = series_x
            testY[test_index] = series_y
            testX2[test_index] = covariate_X
            testY2[test_index] = covariate_Y
        
        else:
            trainX[i-test_n] = series_x
            trainY[i-test_n] = series_y
            trainX2[i-test_n] = covariate_X
            trainY2[i-test_n] = covariate_Y
        # update the start_index
        start_index = start_index - sliding_window_dis
  

    testX_list.append(testX)
    testX2_list.append(testX2)
    testY_list.append(testY)
    testY2_list.append(testY2)
    
    trainX_list.append(trainX)
    trainX2_list.append(trainX2)
    trainY_list.append(trainY)
    trainY2_list.append(trainY2)


testX_dt = np.vstack(testX_list)
testY_dt = np.vstack(testY_list)
testX2_dt = np.vstack(testX2_list)
testY2_dt = np.vstack(testY2_list)

trainX_dt = np.vstack(trainX_list)
trainY_dt = np.vstack(trainY_list)
trainX2_dt = np.vstack(trainX2_list)
trainY2_dt = np.vstack(trainY2_list)

scaler = preprocessing.StandardScaler()
scaler.fit(trainX_dt)

trainX_dt = scaler.transform(trainX_dt)
testX_dt = scaler.transform(testX_dt)
scaler2 = preprocessing.StandardScaler()
scaler2.fit(trainX2[:,:,1])
trainX2[:,:,1] = scaler2.transform(trainX2[:,:,1])
testX2[:,:,1] = scaler2.transform(testX2[:,:,1])

### The filter data
### Select the data of the Nov
isNov = trainX2_dt[:,:,3]>=11
trainX_dt = trainX_dt[isNov[:,0]]
trainY_dt = trainY_dt[isNov[:,0]]
trainX2_dt = trainX2_dt[isNov[:,0]]
trainY2_dt = trainY2_dt[isNov[:,0]]
### Save the data
with open('feature_prepare.pkl', 'wb') as f:
    pickle.dump([trainX_dt,trainX2_dt, trainY_dt,trainY2_dt, testX_dt, testX2_dt,testY_dt,testY2_dt], f, -1)
