## simple TCN without gating and skip connections, with both numeric and categorical features for the past, and with categorical features for the future


import mxnet as mx
from mxnet import gluon,nd,ndarray
from mxnet.gluon import nn

def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)

def _apply_weighting(F, loss, weight=None, sample_weight=None):
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss

class Loss(nn.HybridBlock):
    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError
        
class QuantileLoss(Loss):
    def __init__(self,quantile_alpha=0.5, weight=None, batch_axis=1, **kwargs):
        super(QuantileLoss, self).__init__(weight, batch_axis, **kwargs)
        self.quantile_alpha = quantile_alpha

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        I = pred<=label
        loss = self.quantile_alpha*(label-pred)*I+(1-self.quantile_alpha)*(pred-label)*(1-I)
        #print('loss1',loss)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        #print('loss2',loss)
        return F.sum(loss, axis=self._batch_axis)


class ResidualTCN(nn.Block):
    def __init__(self,d, n_residue=1, k=2,  **kwargs):
        super(ResidualTCN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm()
        
    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nd.relu(out+x[:,:,-out.shape[2]:])
    
    
class ResidualTCN2(nn.Block):
    def __init__(self,d, n_residue=1, k=2,  **kwargs):
        super(ResidualTCN2, self).__init__(**kwargs)
        self.conv1 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nd.relu(out)+x[:,:,-out.shape[2]:]
     

class Residual(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64, flatten=False)
        self.bn1 = nn.BatchNorm(axis=2)
        self.fc2 = nn.Dense(units=xDim, flatten=False)
        self.bn2 = nn.BatchNorm(axis=2)

    def hybrid_forward(self,F, x):
        out = nd.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return nd.relu(out + x) 
    
class futureResidual(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(futureResidual, self).__init__(**kwargs)
        self.fc1 = nn.Dense(32,flatten=False)
        self.bn1 = nn.BatchNorm(axis=2)
        self.fc2 = nn.Dense(units=xDim,flatten=False)
        self.bn2 = nn.BatchNorm(axis=2)
        
    def hybrid_forward(self,F, x_conv, x):
        out = nd.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return nd.relu(x_conv + out)
    
    
    
class TCN(nn.Block):
    def __init__(self, dilation_depth=2, n_repeat=5, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.dilations = [1,2,4,8,16]
        #self.post_res= nn.Sequential()
        self.net=nn.Sequential()
        #self.bn = nn.BatchNorm()
        #self.post_res= nn.Sequential()
        #self.TCN= nn.Sequential()
        with self.name_scope():
            ## The embedding part
            self.conv1 = nn.Conv1D(kernel_size=24, channels=1, activation='relu', strides=2)
            self.conv2 = nn.Conv1D(kernel_size=3, channels=1, activation='relu', strides=2)
            self.pool1 = nn.MaxPool1D(pool_size=3)
            self.store_embedding = nn.Embedding(370,10)
            self.nMonth_embedding = nn.Embedding(12,2)
            self.nYear_embedding = nn.Embedding(3,2)
            self.mDay_embedding = nn.Embedding(31,3)
            self.wday_embedding = nn.Embedding(7,3)
            self.nHour_embedding = nn.Embedding(24,3)
            #self.post_res.add(Residual(xDim=34))
            self.post_res=futureResidual(xDim=14)
            self.net.add(nn.Dense(64, flatten=False))
            self.net.add(nn.BatchNorm(axis=2))
            self.net.add(nn.Activation(activation='relu'))
            self.net.add(nn.Dropout(.2))
            self.net.add(nn.Dense(1,activation='relu',flatten=False))
                       
    def forward(self, x_num, x_cat):
        # preprocess
        embed_concat = nd.concat(
                self.store_embedding(x_cat[:,:,0]),
                x_cat[:,:,1:2],
                self.nYear_embedding(x_cat[:,:,2]),
                self.nMonth_embedding(x_cat[:,:,3]),
                self.mDay_embedding(x_cat[:,:,4]),
                self.wday_embedding(x_cat[:,:,5]),
                self.nHour_embedding(x_cat[:,:,6]), dim=2)
        output = self.preprocess(x_num)
        conv_result = self.pool1(self.conv2(self.conv1(output)))
        #conv_result = conv_result.reshape((conv_result.shape[0], conv_result.shape[2]))
        #output=nd.transpose(output, axes=(0,2,1))
        #output = nd.reshape(output,(output.shape[0], 1,-1))
        #print(output.shape)
        output = nd.broadcast_axis(conv_result, axis=1, size=24)
        post_concat = nd.concat(output, embed_concat, dim=2)
        #output=self.net(self.post_res(post_concat))
        output=self.net(self.post_res(output,embed_concat))
        return output
    
    def residue_forward(self, x, sub_TCN):
        return sub_TCN(x)
    
    def preprocess(self, x):
        ##The shape of the input tensor
        shape0, shape1 = x.shape
        output = x.reshape((shape0, 1, shape1))
        return output
