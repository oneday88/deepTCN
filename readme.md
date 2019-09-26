### Probabilistic Forecasting with Temporal Convolutional Neural Network
Source codes for the paper "probabilistic forecasting with temporal convolutional neural network"
#### Electricity
##### Data preprocessing
   * Download the dataset from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
   * Run "R CMD BATCH electricity/basicPreprocess.R" to generate "modelData.csv" for model training. 
##### Point forecasting
   * python3 electricity/ecPointModel/ec_feature_preprocess.py
   * python3 electricity/ecPointModel/ECPointHuber.py
##### Probabilistic forecasting
   * python3 electricity/NewTCNQuantile/ec_feature_preprocess.py
   * python3 electricity/NewTCNQuantile/ec_probabilistic_forecasting.py
### Traffic
##### Data preprocessing
   * Download the dataset from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/PEMS-SF
   * Run "R CMD BATCH traffic/basicPreprocess.R" to generate "traffic.csv".
   * python3 traffic/traffic_feature_preprocess.py to generate the "tensor_prepare.pkl" for the model training
##### Point forecasting
   * python3 traffic/point/traffic_point_forecasting.py

##### Probabilistic forecasting
### Parts
##### Data preprocessing
##### Probabilistic forecasting

### Reference Paper
[Probabilistic forecasting with temporal convolutional neural network](https://arxiv.org/abs/1906.04397)

KDD 2019 ,Workshop on Mining and Learning from Time Series, 2019

### Kind remind
The total project will be refined in the next months. Also, you can achieve better results if you do better data preprocessing like scaling.
