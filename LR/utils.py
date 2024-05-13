import os
import pandas as pd

def result_dir_make(target_path):
    try:
        if not os.path.exists(target_path):
            print("Make directory %s"%(target_path))
            os.makedirs(target_path)
    except OSError:
        print('Error: Failed to create directory : ' +  target_path)

def data_set_loader(data_file_name):
    data_file_format = data_file_name.split(".")[-1]
    if data_file_format == "csv":
        print("Load CSV File : %s"%(data_file_name))
        return pd.read_csv(data_file_name)
    elif data_file_format == "xlsx":
        print("Load Excel File : %s"%(data_file_name))
        return pd.read_excel(data_file_name)