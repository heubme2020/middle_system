import os
import random

import pandas as pd
from tqdm import tqdm
import shutil
import numpy as np
import time
import threading
from write_stock_data import exchange_reference_dict

def get_csv_files(folder):
    csv_files = []
    for info in os.listdir(folder):
        csv_file = os.path.join(folder, info)
        csv_files.append(csv_file)
    return csv_files


def copy_financial_forecast_train_data():
    exchange_list = ['Taipei', 'Taiwan']
    # exchange_list = ['Taiwan', 'Taipei', 'Kuala', 'Oslo']
    src_folder = 'D:'
    dest_folder = 'financial_daily_data'
    for i in range(len(exchange_list)):
        exchange = exchange_list[i]
        print(exchange)
        exchange_folder = os.path.join(src_folder, exchange, 'financial_forecast_train_data')
        csv_files = get_csv_files(exchange_folder)
        random.seed(1024)
        random.shuffle(csv_files)
        copy_num = int(0.1 * len(csv_files))
        train_files = csv_files[:2*copy_num]
        for i in tqdm(range(len(train_files))):
            train_file = train_files[i]
            train_file_name_new = os.path.basename(train_file)
            train_file_name_new = os.path.join(dest_folder, 'train', train_file_name_new)
            if os.path.exists(train_file_name_new):
                continue
            shutil.copy(train_file, train_file_name_new)
            
        test_files = csv_files[-copy_num:]
        for i in tqdm(range(len(test_files))):
            test_file = test_files[i]
            test_file_name_new = os.path.basename(test_file)
            test_file_name_new = os.path.join(dest_folder, 'test', test_file_name_new)
            if os.path.exists(test_file_name_new):
                continue
            shutil.copy(test_file, test_file_name_new)


def write_growth_death_data(file_list, csv_name):
    if os.path.exists(csv_name) == False:
        basename = os.path.basename(file_list[0])
        basename_splits = basename.split('_')
        symbol = basename_splits[0]
        endDate_splits = basename_splits[1].split('.')
        endDate = float(endDate_splits[0])
        data = pd.read_hdf(file_list[0])
        growth = data['growth'].iloc[-4]
        death = data['death'].iloc[-4]
        input_data = data.iloc[:-4, 1:-2]
        input_data = input_data.T
        input_data = np.ravel(input_data.values)
        data = pd.DataFrame(input_data)
        data = data.T
        data['growth'] = growth
        data['death'] = death
        data.insert(0, 'endDate', endDate)
        data.insert(0, 'symbol', symbol)
        if os.path.exists(csv_name) == False:
            data.to_csv(csv_name, mode='a', header=True, index=False)

    data_17 = pd.read_csv(csv_name)
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        basename = os.path.basename(file)
        basename_splits = basename.split('_')
        symbol = basename_splits[0]
        endDate_splits = basename_splits[1].split('.')
        endDate = float(endDate_splits[0])
        condition = (data_17['symbol'] == symbol) & (data_17['endDate'] == endDate)
        exists = condition.any()
        if exists:
            continue
        else:
            data = pd.read_hdf(file)
            growth = data['growth'].iloc[-4]
            death = data['death'].iloc[-4]
            input_data = data.iloc[:-4, 1:-2]
            input_data = input_data.T
            input_data = np.ravel(input_data.values)
            data = pd.DataFrame(input_data)
            data = data.T
            data['growth'] = growth
            data['death'] = death
            data.insert(0, 'endDate', endDate)
            data.insert(0, 'symbol', symbol)
            data.to_csv(csv_name, mode='a', header=False, index=False)

def copy_growth_death_data(src_folder):
    train_file_list = []
    test_file_list = []

    for key, value in exchange_reference_dict.items():
        try:
            exchange = key
            print(exchange)
            exchange_train_folder = os.path.join(src_folder, exchange, 'growth_death_data/train')
            train_h5_files = [os.path.join(exchange_train_folder, f) for f in os.listdir(exchange_train_folder) if
                              f.endswith('.h5')]
            train_file_list += train_h5_files
            exchange_test_folder = os.path.join(src_folder, exchange, 'growth_death_data/test')
            test_h5_files = [os.path.join(exchange_test_folder, f) for f in os.listdir(exchange_test_folder) if
                             f.endswith('.h5')]
            test_file_list += test_h5_files
        except:
            pass

    random.shuffle(train_file_list)
    train_num = len(train_file_list)
    print(train_num)
    print(len(test_file_list))
    # train_file_list0 = train_file_list[:int(0.5*train_num)]
    # train_file_list1 = train_file_list[int(0.5*train_num):]

    csv_train_file = 'growth_death_train_data_17.csv'
    csv_test_file = 'growth_death_test_data_17.csv'

    write_growth_death_data(train_file_list, csv_train_file)
    write_growth_death_data(test_file_list, csv_test_file)

def copy_growth_death_data_of_a(src_folder):
    train_file_list = []
    test_file_list = []
    try:
        exchange = 'a'
        print(exchange)
        exchange_train_folder = os.path.join(src_folder, exchange, 'growth_death_data/train')
        train_h5_files = [os.path.join(exchange_train_folder, f) for f in os.listdir(exchange_train_folder) if
                          f.endswith('.h5')]
        train_file_list += train_h5_files
        exchange_test_folder = os.path.join(src_folder, exchange, 'growth_death_data/test')
        test_h5_files = [os.path.join(exchange_test_folder, f) for f in os.listdir(exchange_test_folder) if
                         f.endswith('.h5')]
        test_file_list += test_h5_files
    except:
        pass

    random.shuffle(train_file_list)
    train_num = len(train_file_list)
    print(train_num)
    print(len(test_file_list))
    # train_file_list0 = train_file_list[:int(0.5*train_num)]
    # train_file_list1 = train_file_list[int(0.5*train_num):]

    csv_train_file = 'growth_death_train_data_17.csv'
    csv_test_file = 'growth_death_test_data_17.csv'

    write_growth_death_data(train_file_list, csv_train_file)
    write_growth_death_data(test_file_list, csv_test_file)

if __name__ == '__main__':
    # copy_financial_forecast_train_data()
    # copy_growth_and_death_data()
    # copy_growth_death_data('D:')
    # csv_to_parquet('D:/Shenzhen/growth_death_data/test2')
    copy_growth_death_data_of_a('D:')
