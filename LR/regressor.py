import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from tqdm import tqdm
import copy
from FeatureGenerator_LR import term_maker, input_table_maker
from itertools import islice
from sklearn.model_selection import train_test_split, cross_validate, LeaveOneOut, KFold
from sklearn.metrics import accuracy_score
import os


# none : term position : (feature combination info, basic feature combination info)

def regressor_worker(X, y, cv_, fit_intercept=True, do_test=False, test_size=0.2, random_seed=None):
    LR = LogisticRegression(C=10,max_iter=1000,solver='liblinear',random_state=0,fit_intercept=fit_intercept,intercept_scaling=1,class_weight="balanced")
    nfold = cv_

    if do_test==False:
        try:
            LR.fit(X.values, y)
        except:
            pair = []
            for term in X.columns:
                pair.append(term)

            return pair + [np.nan, np.nan, np.nan]

        pair = []
        for i in range(len(X.columns)):
            pair.append(X.columns[i])

        train_score  = LR.score(X.values, y)

        LR = LogisticRegression(C=10,max_iter=1000,solver='liblinear',random_state=0,fit_intercept=fit_intercept,intercept_scaling=1,class_weight="balanced")
        cv_results=cross_validate(LR, X, y, cv=nfold, scoring=['accuracy', 'f1'] ,return_train_score=True)
        cv_train_acc=cv_results['train_accuracy'].mean()
        cv_test_acc=cv_results['test_accuracy'].mean()

        return pair + [train_score, cv_train_acc, cv_test_acc]
    
    if do_test==True:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        try:
            LR.fit(X_train.values, y_train)
        except:
            pair = []
            for term in X_train.columns:
                pair.append(term)
            return pair + [np.nan, np.nan, np.nan, np.nan]

        pair = []
        for i in range(len(X.columns)):
            pair.append(X.columns[i])

        train_score  = LR.score(X_train.values, y_train)
        test_score  = LR.score(X_test.values, y_test)

        LR = LogisticRegression(C=10,max_iter=1000,solver='liblinear',random_state=0,fit_intercept=fit_intercept,intercept_scaling=1,class_weight="balanced")
        cv_results=cross_validate(LR, X, y, cv=nfold, scoring=['accuracy', 'f1'], return_train_score=True)
        cv_train_acc=cv_results['train_accuracy'].mean()
        cv_test_acc=cv_results['test_accuracy'].mean()

        return pair + [train_score, test_score, cv_train_acc, cv_test_acc]



## oprimized version
def regressor_main(data_list, data_key_list, y, part_info, total_case_iteration, dimension, cpu_name, do_test, test_size, cv_, random_seed=None):
    if cv_ in ["loo","LOO","LOOCV","loocv"]:
        cv_info = LeaveOneOut()
    else:
        cv_info = KFold(cv_)


    
    pair = []
    for i in range(dimension):
        pair.append("f%d"%(i))
        
    #result_column = ['intercept_', 'train R2', "train RMSE"]
    
    if do_test == False:
        result_column = ['train ACC', "CV train ACC avg", "CV test ACC avg"]
    else:
        result_column = ['train ACC', 'test ACC', "CV train ACC avg", "CV test ACC avg"]
        
    progress_text = "Process #%s"%cpu_name

    start_num = part_info[0]
    end_num = part_info[1]
    
    temp_memory = []
    temp_form_list = []

    buffer_size = min(10000, end_num-start_num)
    for i in tqdm(range(end_num-start_num), desc=progress_text, position=int(cpu_name)+1, leave=False):
        # 메모리 공간 미리 확보
        if i%buffer_size == 0:
            storage_of_result = np.empty((min(buffer_size, end_num-(start_num+i)), (dimension)+len(result_column)), dtype=np.object_)
            copyed_total_cases_ = copy.copy(total_case_iteration)
            temp_form_list = list(islice(copyed_total_cases_, start_num+i, min(end_num, start_num+i+buffer_size)))
        
        target_form_info = temp_form_list[i%buffer_size]
        terms = []
        for term_info in target_form_info:
            terms.append(term_maker(term_info, data_list, data_key_list))
        input_table = input_table_maker(terms)
        
        storage_of_result[i%buffer_size] = regressor_worker(input_table, y, cv_info, fit_intercept=True, do_test=do_test, test_size=test_size, random_seed=random_seed)
        
        if i%buffer_size == buffer_size-1:
            temp_memory.append(pd.DataFrame(storage_of_result, columns=pair+result_column))
            if len(temp_memory) == 2:
                temp_table = pd.concat([temp_memory[0],temp_memory[1]], axis=0)
                temp_table = temp_table.sort_values(by='CV test ACC avg' ,ascending=False)
                temp_table = temp_table.iloc[:min(len(temp_table), buffer_size), :]
                temp_memory = [temp_table]

        elif i == (end_num-start_num-1):
            temp_memory.append(pd.DataFrame(storage_of_result, columns=pair+result_column))
            temp_table = pd.concat([temp_memory[0],temp_memory[1]], axis=0)
            temp_table = temp_table.sort_values(by='CV test ACC avg' ,ascending=False)
            temp_table = temp_table.iloc[:min(len(temp_table), buffer_size), :]
            
            temp_memory = [temp_table]

    return temp_memory[0]
    
