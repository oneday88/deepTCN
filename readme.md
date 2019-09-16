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
##### Point forecasting
##### Probabilistic forecasting
### Parts
##### Data preprocessing
##### Probabilistic forecasting

### Kind remind
You can achieve better result if do better data preprocessing like scaling.
