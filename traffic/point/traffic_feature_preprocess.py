import numpy as np
import pandas as pd
import _pickle as pickle
from datetime import datetime, timedelta
from sklearn import preprocessing

traffic = pd.read_csv("traffic.csv", header=None,index_col=0).values.T
"""
remove the holiday based on the description of the data set 
"""
dateList = pd.date_range(start = '01/01/2008', end='30/03/2009')
dateRemove  = pd.to_datetime(['2008-01-01', '2008-01-21', '2008-02-18','2008-03-31','2008-03-09', '2008-05-26','2008-07-04','2008-09-01','2008-11-11','2008-11-17', '2008-12-25', '2009-01-01','2009-01-21','2009-02-16',"2009-03-08"])
dateList = [s for s in dateList if s not in dateRemove]

hour_list = []
for nDate in dateList:
    for nHour in range(24):
        tmp_timestamp = nDate+timedelta(hours=nHour)
        hour_list.append(tmp_timestamp)
hour_list = np.array(hour_list)

station_index = list(range(963))

moving_window_dis = 24;
sample_len = 192; #168+24
input_len = 168;
output_len = 24;
total_n = 440-7; ## The total days
test_n = 7     ## The testing days, day of the last 7 days
train_n = total_n - test_n ## The training days

trainX_list = [];trainY_list = [];train_X2_list = []
testX_list = []; testY_list = []; test_X2_list = []


trainX_list = [];trainY_list = [];train_X2_list = []
testX_list = []; testY_list = []; test_X2_list = []
#testX_list = [];testX2_list = [];testY_list = [];testY2_list = []
#for station in station_index:
for station in station_index:
    print(station)
    sub_series = traffic[station,:]

    trainX = np.zeros(shape=(train_n, input_len))       ## The input series
    trainY = np.zeros(shape=(train_n, output_len))      ## The output series

    testX  = np.zeros(shape=(test_n, input_len))        ## The input series
    testY = np.zeros(shape=(test_n, output_len))        ## The output series

    covariate_num = 6   # other features covariate_num: station_id,nYear,nMonth,day_of_month, day_of_week, iHour
    trainX2 = np.zeros(shape=(train_n, sample_len, covariate_num))
    testX2 = np.zeros(shape=(test_n, sample_len, covariate_num))
    ### Testing samples (7+1)*24
    ts_len = sub_series.shape[0]
    start_index = ts_len-sample_len
    for i in range(total_n):
        ### The sequence data
        series_x = sub_series[start_index:start_index+input_len]    #168
        series_y = sub_series[start_index+input_len:start_index+sample_len]  #24
        ### The covariate
        station_XY = np.repeat(station, sample_len)
        ### the time index
        time_index_xy = pd.to_datetime(hour_list[start_index:start_index+sample_len])
        #print(time_index_xy)
        nYear_XY = time_index_xy.year-2008
        nMonth_XY = time_index_xy.month-1
        mDay_XY = time_index_xy.day-1
        wDay_XY = time_index_xy.weekday
        nHour_XY = time_index_xy.hour

        covariate_XY = np.c_[station_XY,nYear_XY,nMonth_XY,mDay_XY,wDay_XY,nHour_XY]  #192*6

        if(i<test_n):
            testX[i] = series_x
            testY[i] = series_y
            testX2[i,:,:] = covariate_XY

        else:
            trainX[i-test_n] = series_x
            trainY[i-test_n] = series_y
            trainX2[i-test_n] = covariate_XY
        # update the start_index
        start_index = start_index - moving_window_dis

    testX_list.append(testX)
    test_X2_list.append(testX2)
    testY_list.append(testY)
    #print(np.array(test_X2_list).shape)
    trainX_list.append(trainX)
    train_X2_list.append(trainX2)
    trainY_list.append(trainY)


trainX_dt = np.vstack(trainX_list)
trainY_dt = np.vstack(trainY_list)
train_X2_dt = np.vstack(train_X2_list)

testX_dt = np.vstack(testX_list)
testY_dt = np.vstack(testY_list)
test_X2_dt = np.vstack(test_X2_list)

### Save the data
with open('tensor_prepare.pkl', 'wb') as f:
    pickle.dump([trainX_dt,train_X2_dt, trainY_dt,testX_dt, test_X2_dt,testY_dt], f, -1)
