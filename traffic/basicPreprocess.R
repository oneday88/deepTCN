library(data.table)

### The fold of traffic dataset
dirPath     <- '/root/Oneday/deeptcn/traffic'

### Load the data
PEMSTrain   <- readLines(file.path(dirPath,'PEMS-SF','PEMS_train'))
PEMSTest    <- readLines(file.path(dirPath,'PEMS-SF','PEMS_test'))
stations    <- readLines(file.path(dirPath,'PEMS-SF', 'stations_list'))
randperm    <- readLines(file.path(dirPath,'PEMS-SF', 'randperm'))

### preprocess of stations and randperm
stations    <- gsub("\\[|\\]", "", stations)
stations    <- as.numeric(unlist(strsplit(stations," ")))

## randperm is the day order
randperm    <- gsub("\\[|\\]", "", randperm)
randperm    <- as.numeric(unlist(strsplit(randperm, " ")))

PEMSData    <- c(PEMSTrain,PEMSTest)
### Every line is a daily record
dayNum      <- length(PEMSData)

trainResult <- lapply(1:dayNum, function(i){
    ##return order of the ith day
    dayIndex    <- which(randperm==i)
    ithRawData     <-  PEMSData[dayIndex]
    seriesList  <- strsplit(ithRawData, split=';')[[1]]
    ### seriesData is the data of one given day:24*963
    seriesData  <- sapply(1:length(seriesList), function(x){
        subSeries   <- as.numeric(unlist(strsplit(seriesList[x]," ")))
        #aggregate to hourly Data
        subSeries   <- matrix(subSeries, nrow=6)
        hourSeries  <- colSums(subSeries)
        hourSeries
    })
    data.table(seriesData)
})

trafficData   <- rbindlist(trainResult)
trafficData[is.na(trafficData)] <- 0
write.table(trafficData, file='traffic.csv', sep=',', col.names=F)
