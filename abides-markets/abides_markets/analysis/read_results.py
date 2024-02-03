import pandas as pd
import pickle
import os


def read_exp_results(results_dir, chosen_keys):
    result_dfs = {}

    for key in chosen_keys:
        result_dfs[key] = []

    for file in os.listdir(results_dir):
        results_dict = pickle.load(open(f"{results_dir}/{file}", "rb"))
        for key, tab in results_dict.items():
            tab['result_file'] = file
            if key in chosen_keys:
                result_dfs[key].append(tab)
    for key, arr in result_dfs.items():
        result_dfs[key] = pd.concat(arr, axis=0)
    return result_dfs


def read_informer_exp_results(results_dir, chosen_keys=['prices', 'volume', 'surplus']):
    result_dfs = read_exp_results(results_dir, chosen_keys)
    for key, df in result_dfs.items():
        df['param_qty'] = df['result_file'].apply(lambda p: p.split("_")[2])
        df['param_order_size'] = df['result_file'].apply(lambda p: p.split("_")[3])
        df['rep_id'] = df['result_file'].apply(lambda p: p.split("_")[5].split(".")[0])
    return result_dfs


def read_informed_flow_exp_results(results_dir, chosen_keys=['prices', 'volume', 'surplus']):
    result_dfs = read_exp_results(results_dir, chosen_keys)
    for key,df in result_dfs.items():
        df['param_qty'] = df['result_file'].apply(lambda p: p.split("_")[3])
        df['param_order_size'] = df['result_file'].apply(lambda p: p.split("_")[4])
        df['param_followers_num'] = df['result_file'].apply(lambda p: p.split("_")[5])
        df['rep_id'] = df['result_file'].apply(lambda p: p.split("_")[7].split(".")[0])
    return result_dfs
