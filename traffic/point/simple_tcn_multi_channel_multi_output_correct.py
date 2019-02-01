## simple TCN with no gating and no skip connections, with only numeric features for the past and with categorical features for the future
## remove relu at last

import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn

class ResidualTCN(nn.Block):
    def __init__(self,d, n_residue=24, k=2,  **kwargs):
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
    def __init__(self,d, n_residue=38, k=2,  **kwargs):
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

    
    
class TCN(nn.Block):
    def __init__(self, dilation_depth=2, n_repeat=5, **kwargs):
        super(TCN, self).__init__(**kwargs)
        self.dilations = [1,2,4,8,16,20,32]
        self.conv_sigmoid = nn.Sequential()
        self.net=nn.Sequential()
        #self.bn = nn.BatchNorm()
        self.post_res= nn.Sequential()
        self.TCN= nn.Sequential()
        with self.name_scope():
            ## The embedding part
            self.id_embedding=nn.Embedding(370,8)
            self.nYear_embedding = nn.Embedding(3,2)
            self.nMonth_embedding = nn.Embedding(12,2)
            self.mDay_embedding = nn.Embedding(31,3)
            self.wday_embedding = nn.Embedding(7,4)
            self.nHour_embedding = nn.Embedding(24,4)
            ## The output part
            for d in self.dilations:
                self.TCN.add(ResidualTCN(d=d))
            self.post_res.add(Residual(xDim=47))
            self.net.add(nn.Dense(64, flatten=False))
            self.net.add(nn.BatchNorm(axis=2))
            self.net.add(nn.Activation(activation='relu'))
            self.net.add(nn.Dropout(.2))
            self.net.add(nn.Dense(1,flatten=False))
            

    def forward(self, x_num, x_cat):
        # preprocess
        embed_concat = nd.concat(
        self.id_embedding(x_cat[:,:,0]),
        self.nYear_embedding(x_cat[:,:,1]),
        self.nMonth_embedding(x_cat[:,:,2]),
        self.mDay_embedding(x_cat[:,:,3]),
        self.wday_embedding(x_cat[:,:,4]),
        self.nHour_embedding(x_cat[:,:,5]), dim=2)
        embed_train = embed_concat[:,0:168,:]
        embed_test = embed_concat[:,168:,:]
        x_num=x_num.reshape(x_num.shape[0],x_num.shape[1],-1)
        conv_x = nd.concat(x_num, embed_train, dim=2)    
        conv_x=nd.transpose(conv_x, axes=(0,2,1))
        output = conv_x
        #skip_connections = []
        for sub_TCN in self.TCN:
            output = self.residue_forward(output, sub_TCN)
            #skip_connections.append(skip)
        #print(skip_connections)
        #output1 = sum([s[:,:,-1] for s in skip_connections]
        output=output[:,:,-1:]
        output=nd.transpose(output, axes=(0,2,1))
        output = nd.broadcast_axis(output , axis=1, size=24)
        post_concat = nd.concat(output, embed_test, dim=2)
        output=self.net(self.post_res(post_concat))
        output=output.reshape(output.shape[0],-1)
        return output
    
    def residue_forward(self, x, sub_TCN):
        return sub_TCN(x)


