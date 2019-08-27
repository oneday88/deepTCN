library(lubridate)
library(data.table)

dirPath     <- '/home/jdroot/Oneday/Order/Electricty'
dt          <- fread(file.path(dirPath,'LD2011_2014.txt'), header=T, sep=';')
timeIndex   <- seq(ymd_hms('2011-01-01 00:00:00'),ymd_hms('2014-12-31 23:00:00'), by = 'hour')

##############################################3
### by aggregating blocks of 4 columns, to obtain T = 26, 304
##############################################3
dt[,V1:=NULL]
aggList <- sapply(dt, function(x){
    x   <- as.numeric(sub(",", ".", x, fixed = TRUE))
    x   <- matrix(x, nrow=4)
    subResult   <- colSums(x)
    subResult
})

timeIndex   <- timeIndex[8761:35064]
aggList     <-  aggList[8761:35064,]
modelData   <-  t(aggList)

write.table(modelData, file='modelData.csv', sep=',', col.names=F)
save(timeIndex, aggList, file='basicPreprocess.rda')     
