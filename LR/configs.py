import configparser
import pandas as pd


config_base = """
[Data Info]
# data file : path of data file.  ex) abc.csv or Dataset/abc.xlsx
data file = 
# index column name of data file : if blank, feature column index will be started from 0
index or excluded column = # ex1) feature1, feature2     ex2) logcon, LiST    ex3) Tc
# y column name of data file : ex) y or Tc etc..
y column = 

[Feature Generation Option]
# BFS combination formation option
# dimension : number of term
# If you want 2D2F(same as 2C2F), set dimension as 2 and set number of combined features as 2
dimension = # With Logistic Regression, put number of to be used features  # ex) 1, 2..
number of combined features = # Fixed to 1 with Logistic Regression # ex) 1, 2 ..

[Calculation Option]
# how many cpu : number of cpu core which will be used to calculation
how many cpu = # Default(blank) is 1
do test = false # default is "false". if you want, change "false" to "true" or "True"
test size = # default is 0.2. If blank, test size = 0.2. It must be a float
random seed = # default is blank. It must be a blank or int
k of kfold cv = # default is 10. It can be int or loo/loocv/LOO/LOOCV

[Result File Option]
# result file : {result root folder}/{result sub folder}/{result name}.{result file format}
result root folder =  # Default = result
result sub folder = # Default is blank. If blank, result file's path will be {result root folder}/{result name}.csv or xlsx
result name =  # Must set this option
# result file format : format of result file. ex) csv or xlsx
result file format = # Default(blank) = csv
"""

def config_loader(config):
    options = dict()
    for section in config.sections():
        for option in config.options(section):
            options[option] = config.get(section, option)
    
    if options['data file'] == '':
        print("Data file is not set up")
        print("Set up data file at Logistic_config.ini")
        return 0

    return options

def config_is_exist():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    try:
        config.read("Logistic_config.ini")
        config.get("Data Info", "data file")
        options = config_loader(config)
        return options
    except:
        with open('Logistic_config.ini', 'w', encoding='utf-8') as configfile:
            configfile.write(config_base)
        print("Config file is created : Logistic_config.ini")
        print("Set options and then run this script")
        return 0




