# Probabilistic Forecasting with Temporal Convolutional Neural Network
This notebook accompanies the paper, "<a href="https://arxiv.org/abs/1906.04397">Probabilistic Forecasting with Temporal Convolutional Neural Network</a>" by Yitian Chen, Yanfei Kang, Yixiong Chen, and Zizuo Wang published at KDD 2019 ,Workshop on Mining and Learning from Time Series

The notebook provides Mxnet codes for the proposed model on the three public datasets, traffic, electricity and parts.

It is worth noting that we use the same model trained on the data before the first prediction window  rather than retraining the model after updating the forecasts.
A rolling-window updating forecasts can acheive higher metrics accuracy.

## parameters
   * dilations (e.g, [1,2,4,8, 16, 32])
   * Loss function: 
      * For point forecasting, Try L1,L2 or Huber Loss
      * For probabilistic forecasting with Quantile regression, you can try quantileLoss with different qunantile point.
      * Users can also construct your loss function based on different distribution assumptions (e.g., Gaussian likelihood)


### Experiments on the traffic dataset
#### Data preprocessing
   * Download the dataset from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/PEMS-SF
   * Run "R CMD BATCH traffic/basicPreprocess.R" to generate "traffic.csv".
   * python3 traffic/trafficModelPrepare.py  to generate the "trafficPrepare.pkl" for the model training.
#### Point  forecasting 
   * python3 traffic/trafficPointHuber.py
#### Probabilistic forecasting based on  quantile regression
   * python3 traffic/trafficQuantileForecast.py
#### Probabilistic forecasting based on Gaussian likelihood
   * python3 traffic/trafficGaussianForecast.py


