#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict# k-Fold cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import time, datetime
import random
from itertools import combinations, permutations
import multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")


##################################### CONFIG###############################################
###########################################################################################
###### Load Data File
### CSV File
# chart = pd.read_csv("Dataset/dataset_0222_RT_rf_Test.csv")
### Excel File
chart = pd.read_excel("Dataset/dataset.xlsx")


###### Set X, y
X_train = chart.iloc[:, 3:] # from (3)+1th column
y_train = chart.iloc[:, 2] # (2)+1th column
X_test=None
y_test=None

###### Calculation Setting
### Random Forest Regressor parameters
n_estimators = 250 # number of trees
max_features = 'sqrt' # {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
random_state = 40 # random seed
criterion = "squared_error" # 

### Number of used features
dimension = 2
### Cross Validation Setting
# LeaveOneOut()
# KFold(10)
###
cross_validation_type = LeaveOneOut()
### Number of parallel processes
n_cpu = 2

###### Result File Setting
### result file : {result_root_folder}/{result_name}_{dimension}F_{cv_info}.csv
result_root_folder = "rf"
result_name = "rf_comb2"

########################################################################################



scoring = ['r2', 'neg_root_mean_squared_error']

###### Result directory making
try:
    if not os.path.exists(result_root_folder):
        os.makedirs(result_root_folder)
except OSError:
    print('Error: Failed to create directory : ' +  result_root_folder)

global symbol
symbol = X_train.columns

def replacer(pair):

    #global symbol
    return ', '.join([symbol[i] for i in pair])

#@jit

def fit_and_scored(x_train, y_train, x_test=None, y_test=None):
    model = RandomForestRegressor(n_estimators = n_estimators, criterion=criterion, max_features=max_features, random_state=random_state)
    model.fit(x_train, y_train)
    
    
    ## Gini (Feature importance)
    gini = model.feature_importances_
    
    #### train set ####
    # prediction
    pred_train = model.predict(x_train)
    # mse
    mse_train = mean_squared_error(y_train, pred_train)
    # r2
    r2_train = r2_score(y_train, pred_train)
    
    #### test set : if (x_test!=None)&&(y_test!=None)####
    if (x_test!=None) and (y_test!=None):
        # prediction
        pred_test = model.predict(x_test)
        # mse
        mse_test = mean_squared_error(y_test, pred_test)
        # r2
        r2_test = r2_score(y_test, pred_test)
        
        return r2_train, mse_train, r2_test, mse_test, gini
    
    return r2_train, mse_train, gini


def cross_validation_score(X, y, cv, scoring=None):
    model = RandomForestRegressor(n_estimators = n_estimators, criterion=criterion, max_features=max_features, random_state=random_state)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    values = dict()
    
    for score_metrix in scoring:
        values['test_%s'%(score_metrix)] = scores["test_%s"%(score_metrix)].mean()
        values['train_%s'%(score_metrix)] = scores["train_%s"%(score_metrix)].mean()
    
    return values


#@jit
def cross_validation_predict(X, y, cv):
    ## Model
    model = RandomForestRegressor(n_estimators = n_estimators, criterion=criterion, max_features=max_features, random_state=random_state)
    ## Cross Validation prediction
    predictions = cross_val_predict(model, X, y, cv=cv)

    ## R2 score of predicted values
    cv_pred_r2 = r2_score(y, predictions)
    cv_pred_rmse = mean_squared_error(y, predictions)**0.5
    
    values = {
              "R2_from_prediction" : cv_pred_r2,
              "RMSE_from_prediction" : cv_pred_rmse,
              "predictions" : predictions,
             }
    
    return values




def parallel_RF(X_train, y_train, X_test, y_test, comb_set, cv, cpu_name=''):
    amount = len(comb_set)

    pair_name_list = np.empty((1, amount), dtype=np.dtype("U%d"%(28*dimension)))
    
    normal_train_r2 = np.empty((amount, 1), dtype=np.float64)
    normal_train_mse = np.empty((amount, 1), dtype=np.float64)
    if X_test!=None and y_test!=None:
        normal_test_r2 = np.empty((amount, 1), dtype=np.float64)
        normal_test_mse = np.empty((amount, 1), dtype=np.float64)
    normal_gini = np.empty((amount, dimension), dtype=np.float64) # len(comb_set[0]) => dimension
    
    cv_train_r2 = np.empty((amount, 1), dtype=np.float64)
    cv_train_rmse = np.empty((amount, 1), dtype=np.float64)
    cv_test_r2 = np.empty((amount, 1), dtype=np.float64)
    cv_test_rmse = np.empty((amount, 1), dtype=np.float64)
    
    cv_r2_from_prediction = np.empty((amount, 1), dtype=np.float64)
    cv_rmse_from_prediction = np.empty((amount, 1), dtype=np.float64)
    cv_predictions = np.empty((amount, len(y_train)))

    progress_text = "Process #%s"%cpu_name
    for i in tqdm(range(amount), desc=progress_text, position=int(cpu_name)+1, leave=False):
        pair = comb_set[i]
        X_use = X_train.iloc[:, pair]
        y_use = y_train

        if (X_test!=None) and (y_test!=None):
            normal_train_r2[i], normal_train_mse[i], normal_test_r2[i], normal_test_mse[i], normal_gini[i] = fit_and_scored(X_use, y_use)
        elif (X_test==None) and (y_test==None):
            normal_train_r2[i], normal_train_mse[i], normal_gini[i] = fit_and_scored(X_use, y_use)
            
        else:
            raise Exception("One of X_test or y_test is not an None")
            
        cv_results = cross_validation_score(X_use, y_use, cv=cv, scoring=scoring)

        cv_train_r2[i] = cv_results['train_r2']
        cv_train_rmse[i] = cv_results['train_neg_root_mean_squared_error']*(-1)
        cv_test_r2[i] = cv_results['test_r2']
        cv_test_rmse[i] = cv_results['test_neg_root_mean_squared_error']*(-1)

        cv_predict = cross_validation_predict(X_use, y_use, cv=cv)

        cv_r2_from_prediction[i] = cv_predict["R2_from_prediction"]
        cv_rmse_from_prediction[i] = cv_predict["RMSE_from_prediction"]
        cv_predictions[i] = cv_predict["predictions"]



        del cv_results, cv_predict

        pair_name_list[0][i] = "%s"%replacer(pair)

    ### result file part
    cv_table = pd.DataFrame()
    cv_table["PAIR"] = pair_name_list[0]
    cv_table['rmse_of_cv_predcition'] = cv_rmse_from_prediction
    cv_table['r2_of_cv_predcition'] = cv_r2_from_prediction
    cv_table['cv_test_rmse_mean'] = cv_test_rmse
    cv_table['cv_test_r2_mean'] = cv_test_r2
    cv_table['cv_train_rmse_mean'] = cv_train_rmse
    cv_table['cv_train_r2_mean'] = cv_train_r2

    normal_table = pd.DataFrame()
    normal_table['R2(normal)_train'] = normal_train_r2.reshape((-1))
    normal_table['MSE(normal)_train'] = normal_train_mse
    if (X_test!=None) and (y_test!=None):
        normal_table['R2(normal)_test'] = normal_test_r2.reshape((-1))
        normal_table['MSE(normal)_test'] = normal_test_mse

    # Gini part
    gini_table_ = pd.DataFrame(columns=list(X_train.columns))
    for i in range(amount):
        form = [np.nan]*len(symbol)
        for j in range(dimension):
            form[comb_set[i][j]] = normal_gini[i][j]
        gini_table_.loc[i] = form

    total_table = pd.concat([cv_table, gini_table_, normal_table], axis=1)


    prediction_table = pd.DataFrame(cv_predictions, columns=chart.iloc[:,0].values)
    prediction_table["PAIR"] = pair_name_list[0]

    reorder1 = prediction_table.columns[-1:].to_list()
    reorder2 = prediction_table.columns[:-1].to_list()
    prediction_table = prediction_table[reorder1+reorder2]

    return [total_table, prediction_table]


def tuple_to_list(cbs):
    for i in range(len(cbs)):
        cbs[i] = list(cbs[i])
    return cbs

def arg_maker(X_train, y_train, X_test, y_test, comb_set, cv, n_cpu, start_time):
    args = []

    amount = len(comb_set)
    for i in range(n_cpu):
        args.append((X_train, y_train, X_test, y_test, comb_set[amount*i//n_cpu:amount*(i+1)//n_cpu], cv, i+1))
        print("arg for P-%d"%(i), end="\r")

    print("arg finished")
    end_time = time.time()
    print("\nStart Delay : %.4fs\n"%(end_time - start_time))
    return args





if __name__ == "__main__":

    start_time = time.time()



    X_train = X_train
    y_train = y_train
    X_test=X_test
    y_test=y_test
    
    symbol = X_train.columns
   
    cv = cross_validation_type
    cv_info = ""
    if str(cv) == "LeaveOneOut()":
        cv_info = "LOOCV"
    else:
        cv_info = "%dfold"%cv.n_splits

    n=n_cpu


    cbs = tuple_to_list(list(combinations(list(range(0,len(X_train.columns))), dimension)))

    total_count = len(cbs)

    print("Total : %10d"%total_count)
    

    mp.freeze_support()
    tqdm.set_lock(mp.RLock())

    pool = mp.Pool(n, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock)
    m = Manager()
    result_list = m.list()


    RF_start = time.time()
    print("\nStart RF with %d process\n"%(n))

    results_ = pool.starmap_async(parallel_RF, arg_maker(X_train, y_train, X_test, y_test, cbs, cv, n, start_time))
    pool.close()
    pool.join()


    results = results_.get()

    result_tables = []
    prediction_tables = []
    for i in range(len(results)):
        result_tables.append(results[i][0])
        prediction_tables.append(results[i][1])

    final_result_df = pd.concat(result_tables)

    final_result_df.to_csv(f'{result_root_folder}/{result_name}_{dimension}F_{cv_info}.csv', index=False, encoding="utf-8-sig")

    final_prediction_df = pd.concat(prediction_tables).set_index("PAIR")
    final_prediction_df.T.to_csv(f'{result_root_folder}/Prediction-{result_name}_{dimension}F_{cv_info}.csv', encoding="utf-8-sig")

    RF_end = time.time()
    elapsed_time = datetime.timedelta(seconds=RF_end-RF_start)
    print("\nElapsed Time - %s\n"%(elapsed_time))

    print(f'Result File: {result_root_folder}/{result_name}_{dimension}F_{cv_info}.csv')
    print(f'Prediction File: {result_root_folder}/Prediction-{result_name}_{dimension}F_{cv_info}.csv\n')






