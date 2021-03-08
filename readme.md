# Probabilistic Forecasting with Temporal Convolutional Neural Network
This notebook accompanies the paper, "<a href="https://arxiv.org/abs/1906.04397"><Probabilistic Forecasting with Temporal Convolutional Neural Network></a>" by Yitian Chen, Yanfei Kang, Yixiong Chen, and Zizhuo Wang published at KDD 2019 ,Workshop on Mining and Learning from Time Series

The notebook provides Mxnet codes for the proposed model on the three public datasets, traffic, electricity and parts.

It is worth noting that we use the same model trained on the data before the first prediction window  rather than retraining the model after updating the forecasts.
A rolling-window updating forecasts can acheive higher metrics accuracy.

## Parameters of deepTCN models
   * inputSize: the length of input sequences. The 'inputSize' should be compatible with the preprocessing codes.
   * outputSize: the length of output sequences.  The 'ouputSize' should be compatible with the preprocessing codes. E.g., in the traffic datasets, we choose inputSize=168, outputSize=24.
   * dilations: dilations of causal convolution nets, this mainly based on the inputSize,  e.g, [1,2,4,8, 16, 32] for the traffic dataset in my implementation.
   * nResidue: we assume the input is a tensor of "batchSize, length, feature-dimension". nResidue is the number of feature-dimensions of the final input.

## Parameters of the trainer
   * Loss function: 
      * For point forecasting, Try L1,L2 or Huber Loss
      * For probabilistic forecasting with Quantile regression, you can try quantileLoss with different qunantile point.
      * Users can also construct your loss function based on different distribution assumptions (e.g., Gaussian likelihood)


### Experiments on the traffic dataset
##### Data preprocessing
   * Download the dataset from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/PEMS-SF
   * Run "R CMD BATCH traffic/basicPreprocess.R" to generate "traffic.csv".
   * python3 traffic/trafficModelPrepare.py  to generate the "trafficPrepare.pkl" for the model training.
##### Point  forecasting 
   * python3 traffic/trafficPointHuber.py
##### Probabilistic forecasting based on  quantile regression
   * python3 traffic/trafficQuantileForecast.py
##### Probabilistic forecasting based on Gaussian likelihood
   * python3 traffic/trafficGaussianForecast.py

### Experiments on the parts dataset
##### Data preprocessing
   * Donwload the dataset from robjhyndman.com: https://robjhyndman.com/expsmooth/expsmooth_data.zip and choose the file "carparts.csv"
   * python3 parts/partsModelPrepare.py to generate "partsPrepare.pkl" for the model training
##### Probabilistic forecasting based on  quantile regression
   * python3 parts/partsQuantileForecast.py
##### Probabilistic forecasting based on Gaussian likelihood
   * python3 parts/partsGaussianForecast.py



