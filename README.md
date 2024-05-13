This is the repository of code and data for paper "Machine Learning Prediction Models for Solid Electrolytes based on Lattice Dynamics Properties"


This codes are based on Python3, and require to install:
  - scikit-learn
  - Pandas with openpyxl
  - tqdm

Explanation
  - 
- LR : Logistic regression code (classification) \
  \
  This directory contains codes to train Logistic regression models.\
  Set the number of features through "Logistic_config.ini" file.\
  All combinations of the number of features trains. following command to run:\
  python main.py 
  
- RF : Random forest regression code (regression) \
  \
  This directory contains codes to train Random forest regression models.\
  Set the initial conditions in the main python code "SE_code_RF_2024.py"\
  , and following command to run:\
  python SE_code_RF_2024.py

Datasets are not included.

Authors
-
This code was primarily written by Donggeon Lee under the guidance of Professor Sooran Kim, and advised by Dr. Jiyeon Kim.

 
  
