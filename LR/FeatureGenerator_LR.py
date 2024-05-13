import numpy as np
import pandas as pd

def basic_feature_generate(feature_data, feature_symbol, include_unit=False):

    dataset_dict = dict()
    
    dataset_dict['{symbol}'.format(symbol="%s"%(feature_symbol))] = np.array(feature_data)


    return dataset_dict #, feature_symbol

def term_maker(term_info, data_list, data_key_list):
    values = 1
    feature_output_name = []
    term_info_len = len(term_info[0])
    for i in range(term_info_len):
        target_feature = data_list[term_info[0][i]]
        target_basic_feature = data_key_list[term_info[0][i]][term_info[1][i]]
        values = values * target_feature[target_basic_feature]
        feature_output_name.append(target_basic_feature)

    return (", ".join(feature_output_name), values)
    

def input_table_maker(terms_):
    input_table = pd.DataFrame()
    for term in terms_:
        input_table[term[0]] = term[1] # term[0] : combination info , term[1] : values

    return input_table