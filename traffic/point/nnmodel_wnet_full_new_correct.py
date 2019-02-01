import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn


class Residual(nn.Block):
    def __init__(self, xDim,  **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64, flatten=False)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim, flatten=False)
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        out = nd.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return nd.relu(out + x)

class Residual2(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(Residual2, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim)
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self,F, x):
        out = nd.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return nd.relu(self.bn2(out + x))

class Residual3(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(Residual3, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim)
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self,F, x):
        out = nd.relu(self.bn1(self.fc1(x)))
        out = nd.relu(self.bn2(self.fc2(out)))
        return (out + x)

class Residual4(nn.HybridBlock):
    def __init__(self, xDim,  **kwargs):
        super(Residual4, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim)
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self,F, x):
        out = nd.relu(self.bn1(self.fc1(nd.relu(x))))
        out = self.bn2(self.fc2(out))
        return (out + x)

class Residual5(nn.Block):
    def __init__(self, xDim,  **kwargs):
        super(Residual5, self).__init__(**kwargs)
        self.fc1 = nn.Dense(64)
        self.bn1 = nn.BatchNorm()
        self.fc2 = nn.Dense(units=xDim)
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        out = self.fc1(nd.relu(self.bn1(x)))
        out = self.fc2(nd.relu(self.bn2(out)))
        return (out + x)

class wavenet(nn.Block):
    def __init__(self,activation='relu', n_residue=24, n_skip= 64, dilation_depth=6, n_repeat=1,**kwargs):
        # the activation test
        assert activation in set(['relu', 'softrelu', 'sigmoid','tanh','selu'])
        super(wavenet, self).__init__(**kwargs)
        ## The dilation convolutional operation
        a = [1,2,4,8,16,32,40,64]
        self.dilation_depth = dilation_depth
        #self.dilations = [4,2,1]
        self.conv_sigmoid = nn.Sequential()
        self.conv_tanh = nn.Sequential()
        self.skip_scale = nn.Sequential()
        self.res_scale = nn.Sequential()
        self.net = nn.Sequential()
        self.outputLayer = nn.Sequential()
        with self.name_scope():
            for d in a:
                self.conv_sigmoid.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=2, dilation=d))
                self.conv_tanh.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=2, dilation=d))
                self.skip_scale.add(nn.Conv1D(in_channels=n_residue, channels=n_skip, kernel_size=1, dilation=d))
                self.res_scale.add(nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=1, dilation=d))
            #self.conv1 = nn.Conv1D(kernel_size=24, channels=1, activation='relu', strides=2)
            #self.conv2 = nn.Conv1D(kernel_size=3, channels=1, activation='relu', strides=2)
            #self.pool1 = nn.MaxPool1D(pool_size=3)
            self.store_embedding = nn.Embedding(370,8)
            self.nMonth_embedding = nn.Embedding(12,2)
            self.nYear_embedding = nn.Embedding(3,2)
            self.mDay_embedding = nn.Embedding(31,3)
            self.wday_embedding = nn.Embedding(7,4)
            self.nHour_embedding = nn.Embedding(24,4)
            self.net.add(Residual(xDim=24))
            self.outputLayer.add(nn.Dense(64, flatten=False))
            self.outputLayer.add(nn.BatchNorm())
            self.outputLayer.add(nn.Activation(activation=activation))
            self.outputLayer.add(nn.Dropout(.2))
            self.outputLayer.add(nn.Dense(1, flatten=False))
        self.conv_post_1 = nn.Conv1D(in_channels=n_skip, channels=n_skip, kernel_size=1)
        self.conv_post_2 = nn.Conv1D(in_channels=n_skip, channels=1, kernel_size=1)

    def forward(self, x_num, x_cat):
        ##residual_output=self.residual(x)
        ##output=self.net(residual_output)
        embed_concat = nd.concat(
                self.store_embedding(x_cat[:,:,0]),
                self.nYear_embedding(x_cat[:,:,1]),
                self.nMonth_embedding(x_cat[:,:,2]),
                self.mDay_embedding(x_cat[:,:,3]),
                self.wday_embedding(x_cat[:,:,4]),
                self.nHour_embedding(x_cat[:,:,5]), dim=2)
        embed_train = embed_concat[:,0:168,:]
        embed_test = embed_concat[:,168:,:]
        x_num=x_num.reshape(x_num.shape[0],x_num.shape[1],-1)
        conv_x = nd.concat(x_num, embed_train, dim=2)    
        conv_x =nd.transpose(conv_x, axes=(0,2,1))  #print(wdays.shape)
        #x_num=x_num.reshape(x_num.shape[0],1,x_num.shape[1])
        #x_concat=nd.concat(x_num,store_id, month, wdays, hour, dim=1)
        output=conv_x
        #print('conv_x:',output.shape)
        skip_connections = []
        for s, t, skip_scale, res_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.res_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, res_scale)
            skip_connections.append(skip)
        #print(skip_connections)
        output_skip = sum([s[:,:,-1] for s in skip_connections])
        output_skip = output_skip.reshape(output_skip.shape[0],output_skip.shape[1], -1 )
        output= self.postprocess(output_skip, embed_test)
        
       # output= output.reshape(output.shape[0], output.shape[2], output.shape[1])
       # output = nd.broadcast_axis(output, axis=1, size=24)
        #embed_result = nd.concat(output, embed_test, dim=2)
        #output=self.outputLayer(self.net(embed_result))
        #output=output.reshape(output.shape[0],-1)
        #print(output1.shape)
        #print(x_concat.shape)
        #conv_output=self.conv(x_concat)
        #print(conv_output.shape)
        #output=self.net(nd.relu(output1))
        #print(output.shape)
        return output
    
    def residue_forward(self, x, conv_sigmoid, conv_tanh, skip_scale, res_scale):
        output = x
        #print(output.shape)
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = nd.sigmoid(output_sigmoid) * nd.tanh(output_tanh)
        #print(output)
        skip = skip_scale(output)
        #print(skip[:,:,-1])
        output = res_scale(output)
        output = output + x[:,:,-output.shape[2]:]
        return output, skip
    
    def postprocess(self, x, embed_test):
        output = nd.relu(x)
        output = self.conv_post_1(output)
        output = nd.relu(output)
        output = self.conv_post_2(output)
        output = nd.broadcast_axis(output , axis=1, size=24)
        embed_result = nd.concat(output, embed_test, dim=2)
        output=self.outputLayer(self.net(embed_result))
        output=output.reshape(output.shape[0],-1)
        return output
