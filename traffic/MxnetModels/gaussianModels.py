import math
import mxnet as mx
from mxnet import gluon,nd,ndarray
from mxnet.gluon import nn

"""
Construct the Gaussian loss for MLE
"""
def GaussianLoss(mu, sigma, y, softZero=1e-4):
    logGaussian =  -nd.log(2*math.pi*(sigma**2)+softZero)/2 -(y- mu)**2/(2*(sigma**2)+softZero)
    res = -nd.sum(logGaussian, axis= 0)
    return nd.mean(res)


"""
The residual blocks of the TCN model: current apply ResidualTCN and futureResidual blocks, users can try other modules
"""
class ResidualTCN(nn.Block):
    def __init__(self,d, n_residue=35, k=2,  **kwargs):
        super(ResidualTCN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm()
        
    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nd.relu(out+x[:,:,-out.shape[2]:])
    
class futureResidual(nn.HybridBlock):
    def __init__(self,xDim=64, **kwargs):
        super(futureResidual, self).__init__(**kwargs)
        self.fc1 = nn.Dense(xDim,flatten=False)
        self.bn1 = nn.BatchNorm(axis=2)
        self.fc2 = nn.Dense(units=xDim,flatten=False)
        self.bn2 = nn.BatchNorm(axis=2)
        
    def hybrid_forward(self,F, lagX, x2):
        out = nd.relu(self.bn1(self.fc1(x2)))
        out = self.bn2(self.fc2(out))
        return nd.relu(nd.concat(lagX,out, dim=2))

"""
The core model
"""
class TCN(nn.Block):
    def __init__(self,inputSize=168, outputSize=24 ,dilations=[1,2,4,8,16,20,32],nResidue=35, actType='relu' ,dropout=0.2, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.dilations = dilations
        self.encoder = nn.Sequential()
        self.outputLayer= nn.Sequential()
        with self.name_scope():
            # The embedding of auxiliary variables
            self.stationEmbedding = nn.Embedding(963,18)
            self.nYearEmbedding = nn.Embedding(3,2)
            self.nMonthEmbedding = nn.Embedding(12,2)
            self.mDayEmbedding = nn.Embedding(31,5)
            self.wdayEmbedding = nn.Embedding(7,3)
            self.nHourEmbedding = nn.Embedding(24,4)
            for d in self.dilations:
                self.encoder.add(ResidualTCN(d=d, n_residue=nResidue))
            self.decoder = (futureResidual(xDim=64))
            self.outputLayer.add(nn.Dense(64, flatten=False))
            self.outputLayer.add(nn.BatchNorm(axis=2))
            self.outputLayer.add(nn.Swish())
            #self.outputLayer.add(nn.Activation(activation=actType))
            self.outputLayer.add(nn.Dropout(dropout))
            #self.outputLayer.add(nn.Dense(1,activation='relu',flatten=False))
            self.mu = nn.Dense(1, flatten=False, activation='relu')
            self.sigma = nn.Dense(1, flatten=False, activation='softrelu')
    
    def forward(self, xNum, xCat):
        # embed the auxiliary variables
        embedConcat = nd.concat(
                self.stationEmbedding(xCat[:,:,0]),
                self.nYearEmbedding(xCat[:,:,1]),
                self.nMonthEmbedding(xCat[:,:,2]),
                self.mDayEmbedding(xCat[:,:,3]),
                self.wdayEmbedding(xCat[:,:,4]),
                self.nHourEmbedding(xCat[:,:,5]),
                             dim=2)
        # The training and testing
        embedTrain = embedConcat[:,0:168,:]
        embedTest = embedConcat[:,168:,:]
        # The input series for encoding
        xNum = xNum.reshape((xNum.shape[0],xNum.shape[1],1))
        inputSeries = nd.concat(xNum, embedTrain, dim=2)
        inputSeries = nd.transpose(inputSeries, axes=(0,2,1))
        for subTCN in self.encoder:
            inputSeries = subTCN(inputSeries)
        # The output 
        output = inputSeries
        output = nd.transpose(output, axes=(0,2,1))
        output = nd.reshape(output,(output.shape[0], 1,-1))
        output = nd.broadcast_axis(output, axis=1, size=self.outputSize)
        # the decoder
        output=self.outputLayer(self.decoder(output, embedTest))
        #output = nd.sum_axis(output, axis=2)
        mu = nd.sum_axis(self.mu(output),  axis=2)
        sigma = nd.sum_axis(self.sigma(output),  axis=2)
        return mu, sigma
