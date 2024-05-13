import copy

def spliter(total_cases_N, n_cpu):
    k, m = divmod(total_cases_N, n_cpu)
    return [(i*k+min(i, m),(i+1)*k+min(i+1, m)) for i in range(n_cpu)]

def arg_maker(data_list, y, total_case_iteration, parts, dimension, n_cpu, do_test=False, test_size=0.2, cv_option_info=10, random_seed=None):
    args = []

    for i in range(n_cpu):
        part_info = copy.deepcopy(parts[i])
        cpu_name = i
        total_case_iteration_copied = copy.deepcopy(total_case_iteration) 
        data_list_copied = [copy.deepcopy(basic_feature) for basic_feature in data_list]
        data_key_list_copied = [list(data.keys()) for data in data_list_copied]
        args.append((data_list_copied, data_key_list_copied, y, part_info, total_case_iteration_copied, dimension, cpu_name, do_test, test_size, cv_option_info, random_seed))
    
    return args

