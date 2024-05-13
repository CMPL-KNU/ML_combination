from configs import config_is_exist
from regressor import regressor_main
from utils import data_set_loader, result_dir_make
from FeatureGenerator_LR import basic_feature_generate
from math import comb as math_comb
from taskManager import spliter, arg_maker
from itertools import product, combinations
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
#import os
#os.environ["MKL_NUM_THREADS"] = "1" 
#os.environ["NUMEXPR_NUM_THREADS"] = "1" 
#os.environ["OMP_NUM_THREADS"] = "1" 
import time, datetime
import warnings
warnings.filterwarnings("ignore")



if __name__=="__main__":

    options = config_is_exist()
    if options == 0:
        print("\nClose")
        raise SystemExit


    # Load data file
    origin = data_set_loader(options['data file'])

    # Separate Index Column
    excluded_columns = options['index or excluded column'].split(",")
    try:
        for remove in excluded_columns:
            idx = origin.pop(remove.strip())
    except:
        idx = origin.pop(origin.columns[0])

    # Separate y(target) Column
    data_y = origin.pop(options['y column'])
    data_X = origin

    print(data_X.columns)
    
    do_test = options["do test"]
    if options["do test"] == "false" or options["do test"] == "False":
        do_test = False
        test_size = 0
        random_seed = None
        
    elif options["do test"] == "True" or options["do test"] == "true":
        do_test = True
        if options["test size"] == "":
            test_size = 0.2
        else:
            test_size = float(options["test size"])

        if options["random seed"] == "":
            random_seed = None
        else:
            random_seed = int(options["random seed"])

    else:
        raise Exception("do test must be a true or false only. Check bfs_config.ini")

    # Set Calculation variables
    given_f_N = len(data_X.columns) # number of input features
    dimension = int(options['dimension'])
    combine_N = 1 # int(options['number of combined features'])
    # conbine_N is fixed to 1 with Logistic Regression


    try:
        n_cpu = int(options['how many cpu'])
    except:
        n_cpu = 1

    # Set result file variables
    result_root_folder = options['result root folder']
    result_sub_folder = options['result sub folder']
    result_name = options['result name']
    result_format = options['result file format']
    if options['result file format'] == "":
        result_format = "csv"

    # Check and make result directory
    if result_sub_folder == '':
        print("Make result file directory")
        result_dir_make(result_root_folder)
        result_file_path = f"{result_root_folder}/{result_name}.{result_format}"
        print(f"Result will be saved to {result_root_folder}/{result_name}.{result_format}")
    else:
        print("Make result file directory")
        result_dir_make(f"{result_root_folder}/{result_sub_folder}")
        result_file_path = f"{result_root_folder}/{result_sub_folder}/{result_name}.{result_format}"
        print(f"Result will be saved to {result_root_folder}/{result_sub_folder}/{result_name}.{result_format}")

    cv_option_info = ''
    try:
        cv_option_info = int(options['k of kfold cv'])
    except:
        if options['k of kfold cv'] == ['']:
            cv_option_info = 10
        else:
            cv_option_info = options['k of kfold cv']

    print(cv_option_info)

    data_list = []
    for i in range(given_f_N):
        target_column = data_X.columns[i]
        data_list.append(basic_feature_generate(data_X[target_column].values, target_column))
    data_key_list = [list(data.keys()) for data in data_list]

    base_f_N = len(data_key_list[0])


    # 항이 하나일 때 가능한 경우의 수
    one_D_n_F_N = (base_f_N**combine_N) * math_comb(given_f_N,combine_N)
    # 전체 경우의 수
    total_cases_N = math_comb(one_D_n_F_N, dimension)

    # print configuration and total cases info
    print("Number of given features : {:}".format(given_f_N))
    #print("Number of Base feature : {:}".format(base_f_N))
    #print("Number of Combine : {:}".format(combine_N))
    print("Number of to be used features : {:}".format(dimension))
    print("\n")
    #print("Possible single term Cases : {:,}".format(one_D_n_F_N))
    print("Total Cases : {:,}".format(total_cases_N))


    feature_comb_cases = list(combinations(list(range(0,given_f_N)), combine_N))
    detailed_comb_cases = list(product(*[list(range(0,base_f_N))]*combine_N))
    cases_ = list(product(*[feature_comb_cases, detailed_comb_cases]))
    total_cases_ = combinations(cases_, dimension)
    


    parts = spliter(total_cases_N, n_cpu)
    args = arg_maker(data_list, data_y, total_cases_, parts, dimension, n_cpu, do_test, test_size, cv_option_info, random_seed)

    mp.freeze_support() # windows OS support
    tqdm.set_lock(mp.RLock())
    pool = mp.Pool(n_cpu, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock)
    
    BFS_start = time.time()

    results_ = pool.starmap_async(regressor_main, args)
    pool.close()
    pool.join()

    BFS_end = time.time()

    elapsed_time = datetime.timedelta(seconds=BFS_end-BFS_start)
    print("\nElapsed Time - %s"%(elapsed_time))

    final_df = pd.concat(results_.get())
    final_df = final_df.sort_values(by='CV test ACC avg', ascending=False).iloc[:min(5000, len(final_df))]
    if result_format == "csv":
        final_df.to_csv(f"{result_file_path}", index=False)
    elif result_format == "xlsx":
        final_df.to_excel(f"{result_file_path}", index=False)
    print(f'Result File : {result_file_path}\n')