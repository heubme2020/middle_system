import os
import ssl
import certifi
import json
import pandas as pd
from urllib.request import urlopen
from sqlalchemy import create_engine, func, update, delete, MetaData, Table, Column, String
from sqlalchemy.orm import sessionmaker
import sqlalchemy as sa
import tushare as ts
import datetime
import csv
import threading
import torch
import torch.nn as nn
import numpy as np
from dateutil.relativedelta import relativedelta
from prophet import Prophet
from tqdm import tqdm

import calendar
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
token ='6263d89930d3ba4b1329d603814270ad'

#exhange_reference_dict
exchange_reference_dict = {'Taiwan':['TW', 'Tai', 'TWSE', 'Taiwan'], 'Taipei':['TWO', 'TWO', 'TPEx', 'Taipei'],
                           'Kuala':['KL', 'KLS', 'KLSE', 'Kuala Lumpur'], 'Warsaw':['WA', 'WSE', 'WSE', 'Warsaw Stock Exchange'],
                           'Vienna':['VI','VIE', 'WBAG', 'Vienna'], 'Lisbon':['LS', 'LIS', 'BVL', 'Lisbon'],
                           'Oslo':['OL', 'OSE', 'OSE', 'Oslo Stock Exchange'], 'Tokyo':['T', 'JPX', 'JPX', 'Tokyo'],
                           'Korea':['KS', 'KSC', 'KSE', 'KSE'], 'Kosdaq':['KQ', 'KOE', 'KOSDAQ', 'KOSDAQ'],
                           'Jakarta':['JK', 'JKT', 'JKSE', 'Jakarta Stock Exchange'], 'Shenzhen':['SZ', 'SHZ', 'SZSE', 'Shenzhen'],
                           'Shanghai':['SS', 'SHH', 'SSE', 'Shanghai']}
#创建数据库
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock')

def get_jsonparsed_data(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_exchange_stock_symbol_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'stock_symbol_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv = df_csv.drop('financialmodelingprep_symbol', axis=1)
    df_csv = df_csv.drop('tushare_symbol', axis=1)
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_income_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'income_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_balance_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'balance_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)

    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_cashflow_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'cashflow_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)

    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_dividend_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'dividend_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()
    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_indicator_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'indicator_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_daily_data(exchange):
    if (os.path.exists(exchange)) == False:
        os.mkdir(exchange)
    table_name = 'daily_' + exchange.lower()
    file_name = exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()
    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def date_back(date_str):
    date_str = str(date_str)
    str_list = list(date_str)    # 字符串转list
    str_list.insert(4, '-')  # 在指定位置插入字符串
    str_list.insert(7, '-')  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

def date_forecast(date_pre):
    date = date_back(date_pre)
    start_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    days_to_add = 20

    date_list = []
    for i in range(days_to_add):
        start_date += datetime.timedelta(days=1)
        while start_date.weekday() >= 5:
            start_date += datetime.timedelta(days=1)
        date_str = start_date.date().strftime("%Y%m%d")
        date_list.append(date_str)
    df = pd.DataFrame({'date': date_list})
    return df

#求取上个季度的最后一天
def get_pre_quarter_end_date(date_str):
    date_str = str(date_str)
    date_check = date_str[-4:]
    if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check =='0930':
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str

def get_mean_std_data(data):
    mean_data = pd.DataFrame()
    std_data = pd.DataFrame()
    date_list = sorted(data['endDate'].unique())
    col_names = []
    for i in tqdm(range(1, len(date_list))):
        date = date_list[i]
        filtered_data = data[data['endDate'] <= date]
        filtered_data = filtered_data.reset_index()
        filtered_data.drop(columns=['index'], inplace=True)
        #如果这个endDate股票数小于100，则放弃
        groups = list(filtered_data.groupby('symbol'))
        if len(groups) < 100:
            continue
        filtered_data = filtered_data.drop('symbol', axis=1)
        col_names = filtered_data.columns.values
        mean_list = [date]
        std_list = [date]
        for k in range(1, len(col_names)):
            col_name = col_names[k]
            mean_value = filtered_data[col_name].mean()
            std_value = filtered_data[col_name].std()
            threshold = 6
            # 根据阈值筛选出异常值的索引
            outlier_indices = filtered_data.index[abs(filtered_data[col_name] - mean_value) > threshold * std_value]
            # print(outlier_indices)
            # 剔除包含异常值的行
            filtered_data_cleaned = filtered_data.drop(outlier_indices)
            filtered_data_cleaned = filtered_data_cleaned.reset_index(drop=True)
            # 计算剔除异常值后的均值和方差
            mean_cleaned = filtered_data_cleaned[col_name].mean()
            std_cleaned = filtered_data_cleaned[col_name].std()
            mean_list.append(mean_cleaned)
            std_list.append(std_cleaned)
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

def get_exchange_data(exchange):
    get_exchange_stock_symbol_data(exchange)
    get_exchange_income_data(exchange)
    get_exchange_balance_data(exchange)
    get_exchange_cashflow_data(exchange)
    get_exchange_dividend_data(exchange)
    get_exchange_indicator_data(exchange)
    get_exchange_daily_data(exchange)

    print('mean&std ...')

    income_data = pd.read_csv(exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(exchange + '/cashflow_' + exchange + '.csv')
    dividend_data = pd.read_csv(exchange + '/dividend_' + exchange + '.csv')
    #合并财务相关数据
    financial_data = pd.merge(dividend_data, income_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    financial_data = financial_data.dropna()
    financial_data = financial_data.reset_index(drop=True)
    financial_data['endDate'] = financial_data['endDate'].apply(get_pre_quarter_end_date)
    financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    financial_data = financial_data.reset_index(drop=True)
    #删除股票数为0的行
    financial_data = financial_data[financial_data['numberOfShares'] != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalStockholdersEquity为0的行
    financial_data = financial_data[financial_data['totalStockholdersEquity'] != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalCurrentLiabilities为0的行
    financial_data = financial_data[financial_data['totalCurrentLiabilities'] != 0]
    financial_data = financial_data.reset_index(drop=True)
    #生成对应的均值，方差矩阵
    mean_data, std_data = get_mean_std_data(financial_data)
    mean_data.to_csv(exchange + '/mean_' + exchange.lower() + '.csv', index=False)
    std_data.to_csv(exchange + '/std_' + exchange.lower() + '.csv', index=False)


def get_data():
    exchange_list = ['Shenzhen', 'Shanghai', 'Taiwan', 'Taipei', 'Tokyo', 'HongKong', 'Korea', 'Seoul', 'NewYork']
    for item in exchange_list:
        print(item)
        get_exchange_data(item)
if __name__ == "__main__":
    # exchange = 'Kuala'
    get_exchange_data('Shenzhen')
    # get_data()
    # get_exchange_dividend_data('Tokyo')




