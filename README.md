This is the repository of code for paper "Machine Learning Prediction Models for Solid Electrolytes based on Lattice Dynamics Properties"


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

- Data \
\
This directory contains data on ionic conductivity, as well as the results from LR and RF models. \
LR_combination_2024.csv : The highest accuracy scores according to the number of features.\
RF_combination_2024.csv : The best prediction perfromance with RMSE and R2 values according to the number of features.\
Ionic_cond_results_2024.csv : A list of calculated materials with ionic conductivity, LR results using the best 6-feature combination,\
 and RF results using the best 5-feature combination.




Authors
-
This code was primarily written by Donggeon Lee under the guidance of Professor [Sooran Kim](https://orcid.org/0000-0001-9568-1838), and advised by Dr. [Jiyeon Kim](https://orcid.org/0000-0001-7088-3871).

 
  
