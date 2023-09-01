import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import os
from sqlalchemy import create_engine, func, update, delete, MetaData, Table, Column, String
from sqlalchemy.orm import sessionmaker
import csv
from dateutil.relativedelta import relativedelta
import datetime
import shutil
import math
from financial_models import Decoder, Transformer
from tqdm import tqdm
import time
import threading
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建数据库
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock')
from torch.utils.data import Dataset, DataLoader

not_standard_list = ['dividend','dividendRatio', 'netAssetValuePerShare', 'pbInverse', 'debtToEquityRatio',
                     'quickRatio', 'earningsMultiple', 'operatingMargin','pretaxProfitMargin', 'netProfitMargin',
                     'revenuePerShare','grossProfitMargin',  'psInverse', 'cashFlowPerShare','cashFlowInverse',
                     'roe', 'dcfPerShare', 'dcfInverse', 'open', 'low', 'high', 'close', 'volume', 'marketCap', 'dayinyear']

def date_back(date_str):
    date_str = str(date_str)
    str_list = list(date_str)  # 字符串转list
    str_list.insert(4, '-')  # 在指定位置插入字符串
    str_list.insert(7, '-')  # 在指定位置插入字符串
    str_out = ''.join(str_list)  # 空字符连接
    return str_out

def day_in_year(date):
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    date = datetime.date(year, month, day)
    idx = int(date.strftime('%j'))
    return idx

def get_mean_std_data(data):
    mean_data =  pd.DataFrame()
    std_data = pd.DataFrame()
    date_list = sorted(data['endDate'].unique())
    col_names = []
    for i in range(1, len(date_list)):
        date = date_list[i]
        filtered_data = data[data['endDate'] <= date]
        filtered_data = filtered_data.reset_index()
        filtered_data.drop(columns=['index'], inplace=True)
        filtered_data = filtered_data.drop('symbol', axis=1)
        col_names = filtered_data.columns.values
        mean_list = [date]
        std_list = [date]
        for k in range(1, len(col_names)):
            col_name = col_names[k]
            mean_value = filtered_data[col_name].mean()
            std_value = filtered_data[col_name].std()
            mean_list.append(mean_value)
            std_list.append(std_value)
        mean_dataframe = pd.DataFrame([mean_list])
        std_dataframe = pd.DataFrame([std_list])
        mean_data = pd.concat([mean_data, mean_dataframe])
        std_data = pd.concat([std_data, std_dataframe])
        mean_data = mean_data.reset_index()
        mean_data.drop(columns=['index'], inplace=True)
        std_data = std_data.reset_index()
        std_data.drop(columns=['index'], inplace=True)
    mean_data.columns = col_names
    std_data.columns = col_names
    return mean_data, std_data


def get_exchange_financial_forecast_train_data(exchange):
    exchange_folder = exchange
    exchange = exchange.lower()
    # exchange_folder = 'D:/' + exchange
    if (os.path.exists(exchange_folder)) == False:
        os.mkdir(exchange_folder)
    # train_data_folder = exchange_folder + '/financial_forecast_train_data'
    train_data_folder = 'financial_daily_data'
    if (os.path.exists(train_data_folder)) == False:
        os.mkdir(train_data_folder)

    daily_data = pd.read_csv(exchange_folder + '/daily_' + exchange + '.csv')
    income_data = pd.read_csv(exchange_folder + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(exchange_folder + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(exchange_folder + '/cashflow_' + exchange + '.csv')
    indicator_data = pd.read_csv(exchange_folder + '/indicator_' + exchange + '.csv')

    #生成对应的均值，方差矩阵
    indicator_mean, indicator_std = get_mean_std_data(indicator_data)
    income_mean, income_std = get_mean_std_data(income_data)
    balance_mean, balance_std = get_mean_std_data(balance_data)
    cashflow_mean, cashflow_std = get_mean_std_data(cashflow_data)
    mean_data = pd.merge(indicator_mean, income_mean,  on=['endDate'])
    mean_data = pd.merge(mean_data, balance_mean, on=['endDate'])
    mean_data = pd.merge(mean_data, cashflow_mean, on=['endDate'])
    std_data = pd.merge(indicator_std, income_std,  on=['endDate'])
    std_data = pd.merge(std_data, balance_std, on=['endDate'])
    std_data = pd.merge(std_data, cashflow_std, on=['endDate'])

    #获得路径下已有的symbol，并删除indicator_data内所有对应数据
    csv_files = [f for f in os.listdir(train_data_folder) if f.endswith('.csv')]
    symbol_list = []
    for i in range(len(csv_files)):
        csv_file = csv_files[i]
        symbol = csv_file.split('_')[0]
        if symbol not in symbol_list:
            symbol_list.append(symbol)
    for i in range(len(symbol_list)):
        symbol = symbol_list[i]
        indicator_data = indicator_data[indicator_data['symbol'] != symbol]
        indicator_data = indicator_data.reset_index()
        indicator_data.drop(columns=['index'], inplace=True)

    groups = list(indicator_data.groupby('symbol'))
    random.shuffle(groups)
    print(len(groups))

    days_input = 768
    days_output = 256

    for i in range(len(groups)):
        print('train_data_generating:' + str(i / (len(groups) + 1.0)))
        group = groups[i][1]
        group = group.reset_index()
        group.drop(columns=['index'], inplace=True)
        group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(group)
        for j in range(sheet_count-1):
            dates = pd.date_range(start=group['endDate'].iloc[j], end=group['endDate'].iloc[j + 1])
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = group.iloc[j].copy()
                data_append['endDate'] = date_str
                group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
        date = group['endDate'].iloc[sheet_count - 1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=group['endDate'].iloc[sheet_count - 1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = group.iloc[sheet_count - 1].copy()
            data_append['endDate'] = date_str
            group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
        group['endDate'] = group['endDate'].apply(lambda x: x.replace('-', ''))
        group.sort_values(by='endDate', inplace=True, ascending=True)
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        group = group.rename(columns={'endDate': 'date'})
        group['date'] = group['date'].apply(lambda x: x.replace('-', ''))
        group['date'] = group['date'].astype(int)
        #合并income表
        income_group = income_data[income_data['symbol'] == groups[i][0]]
        income_group = income_group.reset_index()
        income_group.drop(columns=['index'], inplace=True)
        income_group['endDate'] = income_group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(income_group)
        for j in range(sheet_count - 1):
            dates = pd.date_range(start=income_group['endDate'].iloc[j], end=income_group['endDate'].iloc[j + 1])
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = income_group.iloc[j].copy()
                data_append['endDate'] = date_str
                # income_group = income_group.append(data_append, ignore_index=True)
                income_group = pd.concat([income_group, data_append.to_frame().T], ignore_index=True)
        date = income_group['endDate'].iloc[sheet_count - 1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=income_group['endDate'].iloc[sheet_count - 1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = income_group.iloc[sheet_count - 1].copy()
            data_append['endDate'] = date_str
            income_group = pd.concat([income_group, data_append.to_frame().T], ignore_index=True)
        income_group['endDate'] = income_group['endDate'].apply(lambda x: x.replace('-', ''))
        income_group.sort_values(by='endDate', inplace=True, ascending=True)
        income_group.reset_index(inplace=True)
        income_group.drop(columns='index', inplace=True)
        income_group = income_group.rename(columns={'endDate': 'date'})
        income_group['date'] = income_group['date'].apply(lambda x: x.replace('-', ''))
        income_group['date'] = income_group['date'].astype(int)
        group = pd.merge(group, income_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        #合并balance表
        balance_group = balance_data[balance_data['symbol'] == groups[i][0]]
        balance_group = balance_group.reset_index()
        balance_group.drop(columns=['index'], inplace=True)
        balance_group['endDate'] = balance_group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(balance_group)
        for j in range(sheet_count - 1):
            dates = pd.date_range(start=balance_group['endDate'].iloc[j], end=balance_group['endDate'].iloc[j + 1])
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = balance_group.iloc[j].copy()
                data_append['endDate'] = date_str
                # balance_group = pd.concat([balance_group, data_append], ignore_index=True)
                balance_group = pd.concat([balance_group, data_append.to_frame().T], ignore_index=True)
        date = balance_group['endDate'].iloc[sheet_count - 1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=balance_group['endDate'].iloc[sheet_count - 1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = balance_group.iloc[sheet_count - 1].copy()
            data_append['endDate'] = date_str
            balance_group = pd.concat([balance_group, data_append.to_frame().T], ignore_index=True)
        balance_group['endDate'] = balance_group['endDate'].apply(lambda x: x.replace('-', ''))
        balance_group.sort_values(by='endDate', inplace=True, ascending=True)
        balance_group.reset_index(inplace=True)
        balance_group.drop(columns='index', inplace=True)
        balance_group = balance_group.rename(columns={'endDate': 'date'})
        balance_group['date'] = balance_group['date'].apply(lambda x: x.replace('-', ''))
        balance_group['date'] = balance_group['date'].astype(int)
        group = pd.merge(group, balance_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        #合并cashflow
        cashflow_group = cashflow_data[cashflow_data['symbol'] == groups[i][0]]
        cashflow_group = cashflow_group.reset_index()
        cashflow_group.drop(columns=['index'], inplace=True)
        cashflow_group['endDate'] = cashflow_group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(cashflow_group)
        for j in range(sheet_count - 1):
            dates = pd.date_range(start=cashflow_group['endDate'].iloc[j], end=cashflow_group['endDate'].iloc[j + 1])
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = cashflow_group.iloc[j].copy()
                data_append['endDate'] = date_str
                # cashflow_group = cashflow_group.append(data_append, ignore_index=True)
                cashflow_group = pd.concat([cashflow_group, data_append.to_frame().T], ignore_index=True)
        date = cashflow_group['endDate'].iloc[sheet_count - 1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=cashflow_group['endDate'].iloc[sheet_count - 1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = cashflow_group.iloc[sheet_count - 1].copy()
            data_append['endDate'] = date_str
            cashflow_group = pd.concat([cashflow_group, data_append.to_frame().T], ignore_index=True)
        cashflow_group['endDate'] = cashflow_group['endDate'].apply(lambda x: x.replace('-', ''))
        cashflow_group.sort_values(by='endDate', inplace=True, ascending=True)
        cashflow_group.reset_index(inplace=True)
        cashflow_group.drop(columns='index', inplace=True)
        cashflow_group = cashflow_group.rename(columns={'endDate': 'date'})
        cashflow_group['date'] = cashflow_group['date'].apply(lambda x: x.replace('-', ''))
        cashflow_group['date'] = cashflow_group['date'].astype(int)
        group = pd.merge(group, cashflow_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        #合并daily信息
        daily_group = daily_data[daily_data['symbol'] == groups[i][0]]
        daily_group.reset_index(inplace=True)
        daily_group = daily_group.drop(columns='index')
        group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group = group.drop(columns='index')
        #添加总市值这一列
        group['marketCap'] = group['numberOfShares'] * group['close']
        #添加分红率这一列
        group['dividendRatio'] = group['dividend']/ group['close']
        #添加每股净资产换成市净率的倒数
        group['pbInverse'] = group['netAssetValuePerShare'] / group['close']
        #添加每股销售额换成市销率的倒数
        group['psInverse'] = group['revenuePerShare'] / group['close']
        #每股经营现金流换成对应的，我也不知道叫啥
        group['cashFlowInverse'] = group['cashFlowPerShare']/ group['close']
        #每股的dcf换成对应的
        group['dcfInverse'] = group['dcfPerShare'] / group['close']
        #处理其它几个价格
        group['open'] = group['open'] / group['close']
        group['low'] = group['low'] / group['close']
        group['high'] = group['high'] / group['close']
        #volume换成换手率%
        group['volume'] = group['volume']*group['close']/group['marketCap']
        #加入date在一年中位置这一列
        group['dayinyear'] = group['date']
        group['dayinyear'] = group['dayinyear'].apply(lambda x: day_in_year(x))
        group['dayinyear'] = group['dayinyear']/365.0
        group.drop(columns='symbol', inplace=True)
        # print(group)
        # group.to_csv('group.csv', index=False)
        # #目前必须上市3年后才可以预测
        if len(group) < 1024:
            continue
        else:
            group_data_length = len(group)
            for j in tqdm(range(days_input, group_data_length-days_output)):
                #因为我们只在周五买入，所以只需要周四的数据
                date = group['date'].iloc[j]
                date_object = datetime.datetime.strptime(str(date), "%Y%m%d")
                weekday = date_object.weekday()
                if weekday == 4:
                    fact_data = group.iloc[-256:]
                    fact_data = fact_data.iloc[:, 12:-6]
                    fact_data = fact_data.drop_duplicates(subset=['revenue', 'costOfRevenue', 'capitalExpenditure', 'freeCashFlow'], keep='first')
                    #如果后面的季度信息>4个才有效
                    if fact_data.shape[0] > 4:
                        input_data = group.iloc[:j]
                        train_data = group.iloc[j - days_input:j + days_output]
                        train_data = train_data.reset_index()
                        train_data.drop(columns=['index'], inplace=True)
                        #一花一世界，我们认为用一只股票的所有日期数据做归一化是比较合理的
                        col_names = input_data.columns.values
                        # 单独处理下close和marketCap
                        max_close = input_data['close'].max()
                        min_close = input_data['close'].min()
                        train_data['close'] = (train_data['close'] - min_close)/(max_close - min_close)
                        max_marketCap = input_data['marketCap'].max()
                        min_marketCap = input_data['marketCap'].min()
                        train_data['marketCap'] = (train_data['marketCap'] - min_marketCap)/(max_marketCap - min_marketCap)
                        # 获得上个的季度末日期
                        date_str = datetime.datetime.strptime(str(date), '%Y%m%d').date()
                        quarter_month = ((date_str.month - 1) // 3) * 3 + 1
                        quarter_end_date = datetime.date(date_str.year, quarter_month, 1) + datetime.timedelta(days=-1)
                        endDate = int(quarter_end_date.strftime('%Y%m%d'))

                        for k in range(1, len(input_data.columns)):
                            col_name = col_names[k]
                            #如果不在not_standard_list列表，则进行归一化
                            if col_name not in not_standard_list:
                                mean_value = mean_data.loc[mean_data['endDate'] == endDate, col_name].item()
                                std_value = std_data.loc[std_data['endDate'] == endDate, col_name].item()
                                train_data[col_name] = train_data[col_name] - mean_value
                                if std_value != 0:
                                    train_data[col_name] = train_data[col_name]/std_value
                                else:
                                    train_data[col_name] = 0
                        train_data_name = train_data_folder + '/' + groups[i][0] + '_' + str(date) + '.csv'
                        if (os.path.exists(train_data_name)) == True:
                            continue
                        else:
                            train_data = train_data.fillna(0)
                            train_data.replace([np.inf, -np.inf], 0, inplace=True)
                            # print(train_data)
                            train_data.to_csv(train_data_name, index=False)

# class GRU(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
#         super(GRU, self).__init__()
#         self.num_classes = num_classes  # output size
#         self.num_layers = num_layers  # number of recurrent layers in the GRU
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # neurons in each GRU layer
#         self.device = device
#         # # 加些归一化
#         # self.fc0 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # GRU
#         # GRU model
#         self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
#                           batch_first=True)  # GRU
#         self.fc1 = nn.Linear(hidden_size, num_classes)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         self.fc4 = nn.Linear(hidden_size, num_classes)
#         self.fc5 = nn.Linear(hidden_size, num_classes)
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # hidden state
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # # cell state
#         # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         # propagate input through GRU
#         output, hn = self.GRU(x, h_0)  # (input, hidden, and internal state)
#         # hn = hn.view(-1, self.hidden_size)
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(
#             output), self.fc5(output)
#         # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
#         pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:,
#                                                                                                                 -1, :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out
#
#
# class LSTM(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
#         super(LSTM, self).__init__()
#         self.num_classes = num_classes  # output size
#         self.num_layers = num_layers  # number of recurrent layers in the lstm
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # neurons in each lstm layer
#         self.device = device
#         # # # 加些归一化
#         # self.fc0 = nn.LSTM(input_size, input_size)  # lstm
#         # self.norm = nn.LayerNorm(input_size)  # lstm
#         # LSTM model
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, num_classes)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         self.fc4 = nn.Linear(hidden_size, num_classes)
#         self.fc5 = nn.Linear(hidden_size, num_classes)
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # hidden state
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # cell state
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # propagate input through LSTM
#         output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
#         # hn = hn.view(-1, self.hidden_size)
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(
#             output), self.fc5(output)
#         # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
#         pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:,
#                                                                                                                 -1, :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out
#
#
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, batch_size, device=None):
#         super(Encoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_directions = 1
#         self.batch_size = batch_size
#         self.device = device
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
#
#     def forward(self, x):
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         output, (h, c) = self.lstm(x, (h_0, c_0))
#         return output, h, c
#
#
# class Decoder(nn.Module):
#     def __init__(self, input_length, hidden_size, num_layers=1, batch_size=1024, device=None):
#         super(Decoder, self).__init__()
#         self.input_length = input_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.input_size = hidden_size
#         self.batch_size = batch_size
#         self.device = device
#
#         self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
#                             batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, 1)
#         self.fc2 = nn.Linear(hidden_size, 1)
#         self.fc3 = nn.Linear(hidden_size, 1)
#         self.fc4 = nn.Linear(hidden_size, 1)
#         self.fc5 = nn.Linear(hidden_size, 1)
#
#     def forward(self, x, hn, cn):
#         # x = x.view(self.batch_size, 1, self.input_size)
#         lstm_out, (hn, cn) = self.lstm(x, (hn, cn))
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(hn), self.fc2(hn), self.fc3(hn), self.fc4(hn), self.fc5(hn)
#         pred1, pred2, pred3, pred4, pred5 = pred1[-1, :, :], pred2[-1, :, :], pred3[-1, :, :], pred4[-1, :, :], pred5[
#                                                                                                                 -1, :,
#                                                                                                                 :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out, hn, cn
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024,
#                  device=None):
#         super(Seq2Seq, self).__init__()
#         self.input_size = input_size
#         self.input_length = input_length
#         self.output_length = output_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.device = device
#
#         self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#         self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#
#     def forward(self, x):
#         encoder_output, hidden, cell = self.encoder(x)
#         batch_length = encoder_output.shape[0]
#         encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
#         outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
#         for day_idx in range(self.output_length):
#             print(encoder_output.shape)
#             print(hidden.shape)
#             out, hidden, cell = self.decoder(encoder_output, hidden, cell)
#             outputs[:, :, day_idx] = out[:, :, -1]
#         return outputs
#
#
# class Attention(nn.Module):
#     def __init__(self, input_hid_dim, output_hid_dim):
#         super().__init__()
#         self.attention = nn.Linear(input_hid_dim + output_hid_dim, input_hid_dim)
#         self.align = nn.Linear(output_hid_dim, 1)
#
#     def forward(self, hidden, encoder_outputs):
#         batch_size = encoder_outputs.shape[0]
#         src_len = encoder_outputs.shape[1]
#         feature_size = encoder_outputs.shape[2]
#         hidden = hidden.permute(1, 0, 2)
#         hidden = hidden.repeat(1, src_len, 1)
#         attention_outputs = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
#         align_score = self.align(attention_outputs).squeeze(2)
#         attention_weights_days = F.softmax(align_score, dim=1).unsqueeze(2)
#         attention_weights_days = attention_weights_days.repeat(1, 1, feature_size)
#         attention_weight_features = F.softmax(attention_outputs, dim=2)
#         attention_weights = attention_weight_features.mul(attention_weights_days)
#         return attention_weights
#
#
# class Seq2SeqWithAttention(nn.Module):
#     def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024,
#                  device=None):
#         super(Seq2SeqWithAttention, self).__init__()
#         self.input_size = input_size
#         self.input_length = input_length
#         self.output_length = output_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.device = device
#
#         self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#         self.attention = Attention(self.input_size, self.hidden_size).to(self.device)
#         self.vector = nn.Linear(input_size + input_size, input_size)
#         self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#
#     def forward(self, x):
#         encoder_output, hidden, cell = self.encoder(x)
#         batch_length = encoder_output.shape[0]
#         encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
#         outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
#         for day_idx in range(self.output_length):
#             # print(encoder_output.shape)
#             # print(hidden.shape)
#             attention_weights = self.attention(hidden, encoder_output)
#             vector_output = self.vector(torch.cat((encoder_output, attention_weights), dim=2))
#             out, hidden, cell = self.decoder(vector_output, hidden, cell)
#             outputs[:, :, day_idx] = out[:, :, -1]
#         return outputs


class CSV_Dataset(Dataset):
    def __init__(self, csv_files, device=None):
        self.csv_files = csv_files
        self.device = device

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        file = self.csv_files[idx]
        data = pd.read_csv(file)
        input_data = data.iloc[:-256]
        input_data = input_data.iloc[:, 1:]
        fact_data = data.iloc[-256:]
        fact_data = fact_data.iloc[:, 15:-12]
        fact_data = fact_data.drop_duplicates(subset=['revenue','costOfRevenue','capitalExpenditure','freeCashFlow'], keep='first')
        if fact_data.shape[0] > 4:
            fact_data = fact_data.iloc[1:5]
        elif fact_data.shape[0] < 4:
            repeat_num = 4 - fact_data.shape[0]
            for i in range(repeat_num):
                fact_data.loc[len(fact_data)] = fact_data.iloc[-1]
        if self.device is None:
            input_data = input_data.values
            fact_data = fact_data.values
        else:
            input_data = input_data.values
            fact_data = fact_data.values
            input_data = torch.tensor(input_data, device=self.device)
            fact_data = torch.tensor(fact_data, device=self.device)
        return input_data, fact_data

def median_squared_error(output, target):
    output_median = torch.median(output, dim=1).values
    target_median = torch.median(target, dim=1).values
    loss = torch.mean(((output_median / target_median - 1.0) ** 2))
    return loss

def max_squared_error(output, target):
    output_max = torch.max(output, dim=1).values
    target_max = torch.max(target, dim=1).values
    loss = torch.mean(((output_max / target_max - 1.0) ** 2))
    return loss

def weight_mse_loss(output, target):
    output = torch.squeeze(output)
    target = torch.squeeze(target)
    loss_0 = torch.mean(0.4 * (output[:, 0] - target[:, 0]) ** 2)
    loss_1 = torch.mean(0.3 * (output[:, 1] - target[:, 1]) ** 2)
    loss_2 = torch.mean(0.2 * (output[:, 2] - target[:, 2]) ** 2)
    loss_3 = torch.mean(0.1 * (output[:, 3] - target[:, 3]) ** 2)
    loss = loss_0 + loss_1 + loss_2 + loss_3
    return loss


def training_loop(n_epochs, model, optimiser, loss_fn, train_loader, validate_loader, model_name):
    validate_length = len(validate_loader)
    min_validate_loss = float("inf")
    for epoch in range(n_epochs):
        mean_loss = 0
        step_num = 0
        for i, data in enumerate(train_loader):
            input_data, fact_data = data
            input_data = input_data.float()
            fact_data = fact_data.float()
            output_data = model(input_data)
            optimiser.zero_grad()  # calculate the gradient, manually setting to 0
            loss = 0
            for k in range(fact_data.shape[2]):
                loss_k = weight_mse_loss(output_data[k, :, :], fact_data[:, :, k])
                loss = loss + loss_k
            mean_loss = (mean_loss*step_num + loss)/float(step_num + 1)
            step_num = step_num + 1
            loss.backward()  # calculates the loss of the loss function
            optimiser.step()  # improve from loss, i.e backprop
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min loss: %1.5f" % (epoch, loss.item(), mean_loss,  min_validate_loss))

        # 验证集部分
        total_validate_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validate_loader):
                input_data, fact_data = data
                input_data = input_data.float()
                fact_data = fact_data.float()
                output_data = model(input_data)
                loss = 0
                for k in range(fact_data.shape[2]):
                    loss_k = weight_mse_loss(output_data[k, :, :], fact_data[:, :, k])
                    loss = loss + loss_k
                total_validate_loss = total_validate_loss + loss
        total_validate_loss = total_validate_loss/float(validate_length)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, total_validate_loss))
        if total_validate_loss < min_validate_loss:
            min_validate_loss = total_validate_loss
            torch.save(model, model_name)


def forecast_plt(model_path, exchange):
    device = torch.device(0)
    exchange = exchange.lower()
    model = torch.load(model_path)
    print(model.eval())
    data_name = exchange + '.csv'
    data = pd.read_csv(data_name)
    data = data[['date', 'open', 'low', 'high', 'close', 'volume']]
    test_data = data.iloc[:522]
    test_data = test_data.reset_index()
    test_data.drop(columns=['index'], inplace=True)
    test_data.sort_values(by='date', inplace=True, ascending=True)
    test_data = test_data.reset_index()
    test_data.drop(columns=['index'], inplace=True)
    test_input = test_data.iloc[:512]
    test_output = test_data.iloc[512:]
    print(test_input)
    test_input = test_input[['open', 'low', 'high', 'close', 'volume']]
    mean = test_input.mean(axis=0)
    std = test_input.std(axis=0)
    print(mean)
    print(std)
    test_input = test_input.sub(mean, axis=1)
    print(test_input)
    test_input = test_input.div(std, axis=1)
    back_input = test_input.mul(std, axis=1)
    back_input = back_input.add(mean, axis=1)
    print(back_input)
    print(test_input)
    test_output = test_data.iloc[-10:]
    test_input = test_input.values
    test_input = torch.tensor(test_input)
    test_input = test_input.float()
    test_input = test_input.reshape(1, 512, 5)
    test_input = torch.tensor(test_input, device=device)
    yhat = model(test_input)
    yhat = yhat.cpu()
    yhat = yhat.detach().numpy()
    yhat = pd.DataFrame(yhat[:, 0, :])
    # yhat = yhat.values
    print(yhat)
    # print(yhat.shape)
    yhat = yhat.T
    # print(yhat)
    #
    # # yhat = pd.DataFrame(yhat)
    # yhat = yhat.values
    # print(yhat)
    # yhat = pd.DataFrame(yhat)
    yhat.columns = [['open', 'low', 'high', 'close', 'volume']]
    yhat.reset_index(inplace=True)
    yhat.drop(columns=['index'], inplace=True)
    print(yhat)
    # # yhat = yhat.astype('float64')
    # # print(yhat.info())
    print(std)
    yhat['open'] = yhat['open'] * std['open']
    yhat['low'] = yhat['low'] * std['low']
    yhat['high'] = yhat['high'] * std['high']
    yhat['close'] = yhat['close'] * std['close']
    yhat['volume'] = yhat['volume'] * std['volume']
    yhat['open'] = yhat['open'] + mean['open']
    yhat['low'] = yhat['low'] + mean['low']
    yhat['high'] = yhat['high'] + mean['high']
    yhat['close'] = yhat['close'] + mean['close']
    yhat['volume'] = yhat['volume'] + mean['volume']
    # yhat = yhat.mul(std, axis=1)
    print(yhat)
    print(test_output)
    # yhat = yhat + mean
    # print(yhat)


def get_csv_files(folder):
    csv_files = []
    for info in os.listdir(folder):
        csv_file = os.path.join(folder, info)
        csv_files.append(csv_file)
    return csv_files


def copy_csv_files(csv_files, ratio):
    random.shuffle(csv_files)
    copy_num = int(ratio * len(csv_files))
    train_files = csv_files[:copy_num]
    for i in tqdm(range(len(train_files))):
        train_file = train_files[i]
        train_file_name_new = os.path.basename(train_file)
        train_file_name_new = os.path.join('C:/Users/86155/PycharmProjects/Stock/financial_daily_data/',
                                           train_file_name_new)
        shutil.copy(train_file, train_file_name_new)

def train_model():
    device = torch.device(0)
    n_epochs = 100  # 1000 epochs
    learning_rate = 0.001  # 0.001 lr
    input_size = 125  # number of features
    hidden_size = 16  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers
    batch_size = 64
    num_classes = 4  # number of output classes
    input_length = 768
    model_name = 'decoder_financial_forcast.pt'
    model = Decoder(input_size, input_length, device)
    if os.path.exists(model_name) == True:
        model = torch.load(model_name)
    model = model.to(device)
    # model = model.float()
    # loss_fn = weight_mse_loss()
    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    # loss_fn = torch.nn.L1Loss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    train_folder = 'financial_daily_data'
    csv_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    print(len(csv_files))
    random.shuffle(csv_files)
    ratio = 0.1
    validate_num = int(ratio * len(csv_files))
    train_files = csv_files[:-validate_num]
    validate_files = csv_files[-validate_num:]
    train_dataset = CSV_Dataset(train_files, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validate_dataset = CSV_Dataset(validate_files, device=device)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True)

    training_loop(n_epochs=n_epochs,
                  model=model,
                  optimiser=optimiser,
                  loss_fn=loss_fn,
                  train_loader=train_loader,
                  validate_loader=validate_loader,
                  model_name=model_name)
def copy_data(ratio):
    exchange_list = ['Taiwan', 'Taipei']
    # exchange_list = ['Taiwan', 'Taipei', 'Kuala', 'Oslo']
    for i in range(len(exchange_list)):
        exchange = exchange_list[i]
        print(exchange)
        exchange_folder = 'D:/' + exchange + '/financial_forecast_train_data'
        csv_files = get_csv_files(exchange_folder)
        copy_csv_files(csv_files, ratio)

if __name__ == '__main__':
    train_model()
    # forecast_plt('Bitcoin_daily_forcast.pt', 'Bitcoin')
    # train_model()
    # get_exchange_daily_data('Taipei')
    # get_exchange_daily_forecast_train_data('Shenzhen')
    # data = pd.read_csv('Taipei/daily_forecast_train_data/3551.TPEx_20151029.csv')
    # print(data)
    # data = data.iloc[:,1:]
    # print(data)
    # data = data.iloc[:,-5:]
    # print(data)
    # get_exchange_daily_forecast_train_data('Jakarta')
    # get_exchange_financial_forecast_train_data('Taipei')
    # get_exchange_financial_forecast_train_data('Oslo')
    # copy_data(0.1)
