import os
import random
import ssl
import certifi
import json
import pandas as pd
from urllib.request import urlopen
from sqlalchemy import create_engine, func, update, delete, MetaData, Table, Column, String, Integer, Text, insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sa
import tushare as ts
import datetime
import csv
import threading
import numpy as np
from prophet import Prophet
from get_stock_data import get_exchange_data
from tqdm import tqdm
import multiprocessing
from datetime import datetime, timedelta
import datetime as dt
import time
from sqlalchemy import inspect

# from bs4 import BeautifulSoup
# -*- coding: utf-8 -*-
'''
数据查询接口
'''
import json
import requests
from pandas import json_normalize

lxr_token = '646143fb-b1ca-40c9-b73d-f47b1dfbc062'
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock', pool_size=10, max_overflow=20)
pd.set_option('display.max_columns', None)

def get_all_symbols():
    datas = {
        "token": lxr_token
    }
    r = requests.post('https://open.lixinger.com/api/cn/company', json=datas)
    df_company = json_normalize(r.json()['data'])
    return df_company

def write_exchange_stock_symbol_data():
    exchange = 'a'
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    if exchange_stock_symbol_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        drop_table = Table(exchange_stock_symbol_name, metadata, autoload=True, autoload_with=engine)
        drop_table.drop(engine)
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(exchange_stock_symbol_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('name', String(200)),
                      Column('exchange', String(50)),
                      Column('ipoDate', String(50)),
                      Column('fsType', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        data = get_all_symbols()
        symbol_data = data['stockCode']
        data.insert(0, 'symbol', symbol_data)
        data = data.drop('market', axis=1)
        data = data.drop('areaCode', axis=1)
        data = data.drop('stockCode', axis=1)
        data = data.drop('listingStatus', axis=1)
        data = data.drop('mutualMarkets', axis=1)
        data = data.dropna(subset=['ipoDate'])
        data = data.reset_index(drop=True)
        data['ipoDate'] = pd.to_datetime(data['ipoDate'])
        data['ipoDate'] = data['ipoDate'].dt.strftime('%Y%m%d')
        data['symbol'] = data['symbol'].str.cat(data['exchange'], sep='.')
        data.to_sql(exchange_stock_symbol_name, con=engine, if_exists='append', index=False)
        print(exchange_stock_symbol_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(exchange_stock_symbol_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('name', String(200)),
                      Column('exchange', String(50)),
                      Column('ipoDate', String(50)),
                      Column('fsType', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        data = get_all_symbols()
        symbol_data = data['stockCode']
        data.insert(0, 'symbol', symbol_data)
        data = data.drop('market', axis=1)
        data = data.drop('areaCode', axis=1)
        data = data.drop('stockCode', axis=1)
        data = data.drop('listingStatus', axis=1)
        data = data.drop('mutualMarkets', axis=1)
        data = data.dropna(subset=['ipoDate'])
        data = data.reset_index(drop=True)
        data['ipoDate'] = pd.to_datetime(data['ipoDate'])
        data['ipoDate'] = data['ipoDate'].dt.strftime('%Y%m%d')
        data['symbol'] = data['symbol'].str.cat(data['exchange'], sep='.')
        data.to_sql(exchange_stock_symbol_name, con=engine, if_exists='append', index=False)
        print(exchange_stock_symbol_name + ' data inited!')

def get_symbol_daily_data(symbol_full, start_date):
    symbol = symbol_full.split('.')[0]
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    ten_years_later = start_date + timedelta(days=3652)
    ten_years_later = ten_years_later.strftime("%Y-%m-%d")
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    start_date_f = start_date.strftime("%Y-%m-%d")
    end_date_f = ten_years_later
    daily_data = pd.DataFrame()
    while True:
        if start_date_f >= today:
            break
        if end_date_f > today:
            end_date_f = today
        datas_ex = {
            "token": lxr_token,
            "type": "ex_rights",
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCode": symbol
        }
        r_ex = requests.post('https://open.lixinger.com/api/cn/company/candlestick', json=datas_ex)
        df_ex = json_normalize(r_ex.json()['data'])

        datas_fc = {
            "token": lxr_token,
            "type": "fc_rights",
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCode": symbol
        }
        r_fc = requests.post('https://open.lixinger.com/api/cn/company/candlestick', json=datas_fc)
        df_fc = json_normalize(r_fc.json()['data'])
        df_fc = df_fc.rename(columns={'close': 'adjClose'})
        df_fc = df_fc.drop('open', axis=1)
        df_fc = df_fc.drop('low', axis=1)
        df_fc = df_fc.drop('high', axis=1)
        df_fc = df_fc.drop('volume', axis=1)
        df_fc = df_fc.drop('amount', axis=1)
        df_fc = df_fc.drop('change', axis=1)
        df = pd.merge(df_ex, df_fc)
        start_date_f = end_date_f
        ten_years_later = datetime.strptime(start_date_f, "%Y-%m-%d") + timedelta(days=3652)
        ten_years_later = ten_years_later.strftime("%Y-%m-%d")
        end_date_f = ten_years_later
        daily_data = pd.concat([daily_data, df], ignore_index=True)
    daily_data = daily_data.drop_duplicates(subset='date')
    daily_data = daily_data.sort_values(by='date')
    daily_data = daily_data.reset_index(drop=True)
    daily_data.insert(0, 'symbol', symbol_full)
    daily_data = daily_data.drop('amount', axis=1)
    daily_data = daily_data.drop('change', axis=1)
    daily_data['date'] = daily_data['date'].str.replace('-', '')
    new_order = ['symbol', 'date', 'open', 'low', 'high', 'close', 'adjClose', 'volume']
    daily_data = daily_data[new_order]
    return daily_data

def write_daily_data_into_table(stock_symbol_data):
    daily_exchange_name = 'daily_a'
    #建立连接
    metadata = MetaData()
    daily_table = Table(daily_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(daily_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        ipo_date = str(stock_data['ipoDate'])
        if ipo_date == '':
            ipo_date = '19000101'
        ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
        last_date = conn.execute(f"SELECT MAX(date) FROM {daily_exchange_name} WHERE symbol = '{symbol}'")
        last_date = last_date.scalar()
        if last_date is None:
            last_date = ipo_date
        else:
            last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
        date_format = "%Y-%m-%d"
        last_date = datetime.strptime(last_date, date_format)
        last_date = last_date + timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        try:
            daily_data = get_symbol_daily_data(symbol, last_date)
        except:
            continue
        insert_num = 0
        for index, row in daily_data.iterrows():
            data = row.to_dict()
            try:
                conn.execute(daily_table.insert(), data)
                insert_num += 1
            except:
                pass
        print('insert_num:' + str(insert_num))


def write_exchange_daily_data():
    exchange = 'a'
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    daily_exchange_name = 'daily_' + exchange
    if daily_exchange_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        # 如果表格存在，则输出提示信息
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入daily_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(daily_exchange_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(daily_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('date', String(50), primary_key=True),
                      Column('open', String(50)),
                      Column('low', String(50)),
                      Column('high', String(50)),
                      Column('close', String(50)),
                      Column('adjClose', String(50)),
                      Column('volume', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入daily_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(daily_exchange_name + ' data inited!')

#求取上个季度的最后一天
def get_pre_quarter_end_date(date_str):
    if '-' in date_str:
        date_str = date_str.replace('-', '')
    date = datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = dt.date(date.year, quarter_month, 1) + timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str

def get_quarter_end_date(date_str):
    if '-' in date_str:
        date_str = date_str.replace('-', '')
    date = datetime.strptime(date_str, '%Y%m%d').date()
    current_quarter = (date.month - 1) // 3 + 1  # 计算当前季度
    first_day_of_quarter = dt.date(date.year, (current_quarter - 1) * 3 + 1, 1)
    # 计算下一季度的第一天
    if current_quarter < 4:
        next_quarter = current_quarter + 1
        next_year = date.year
    else:
        next_quarter = 1
        next_year = date.year + 1
    first_day_of_next_quarter = dt.date(next_year, (next_quarter - 1) * 3 + 1, 1)
    last_day_of_quarter = first_day_of_next_quarter - timedelta(days=1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = last_day_of_quarter.strftime('%Y%m%d')
    return quarter_end_date_str

def format_date(date_str):
    # 将日期时间字符串解析为日期时间对象
    date_obj = pd.to_datetime(date_str)
    # 将日期时间对象格式化为 "YYYY-MM-DD" 格式
    return date_obj.strftime("%Y-%m-%d")

def get_symbol_dividend_data(symbol_full, start_date):
    symbol = symbol_full.split('.')[0]
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    ten_years_later = start_date + timedelta(days=3652)
    ten_years_later = ten_years_later.strftime("%Y-%m-%d")
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    start_date_f = start_date.strftime("%Y-%m-%d")
    end_date_f = ten_years_later
    equity_data = pd.DataFrame()
    dividend_data = pd.DataFrame()
    while True:
        if start_date_f >= today:
            break
        if end_date_f > today:
            end_date_f = today
        datas_equity = {
            "token": lxr_token,
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCode": symbol
        }
        r_equity = requests.post('https://open.lixinger.com/api/cn/company/equity-change', json=datas_equity)
        df_equity = json_normalize(r_equity.json()['data'])
        df_equity = df_equity.drop('changeReason', axis=1)
        df_equity = df_equity.rename(columns={'capitalization': 'numberOfShares'})
        df_equity['date'] = df_equity['date'].apply(format_date)
        df_equity['date'] = df_equity['date'].str.replace('-', '')
        df_equity['date'] = df_equity['date'].apply(get_quarter_end_date)
        df_equity['date'] = pd.to_datetime(df_equity['date'], format='%Y%m%d')
        df_equity = df_equity[~df_equity['date'].duplicated(keep='first')]
        df_equity.set_index('date', inplace=True)
        df_equity = df_equity.resample('Q').fillna(method='ffill')
        df_equity.reset_index(inplace=True)
        df_equity['date'] = df_equity['date'].astype(str)
        df_equity['date'] = df_equity['date'].str.replace('-', '')

        datas_dividend = {
            "token": lxr_token,
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCode": symbol
        }
        r_dividend = requests.post('https://open.lixinger.com/api/cn/company/dividend', json=datas_dividend)
        df_dividend = json_normalize(r_dividend.json()['data'])
        df_dividend['date'] = df_dividend['date'].apply(format_date)
        df_dividend['date'] = df_dividend['date'].str.replace('-', '')
        df_dividend = df_dividend.drop('currency', axis=1)
        df_dividend = df_dividend.drop('paymentDate', axis=1)
        df_dividend = df_dividend.drop('exDate', axis=1)
        df_dividend = df_dividend.drop('registerDate', axis=1)
        df_dividend = df_dividend.drop('status', axis=1)
        df_dividend = df_dividend.drop('dividendAmount', axis=1)
        df_dividend['date'] = df_dividend['date'].apply(get_pre_quarter_end_date)
        df_dividend['date'] = pd.to_datetime(df_dividend['date'], format='%Y%m%d')
        df_dividend = df_dividend[~df_dividend['date'].duplicated(keep='first')]
        df_dividend.set_index('date', inplace=True)
        df_dividend = df_dividend.resample('Q').fillna(method='ffill')
        df_dividend.reset_index(inplace=True)
        df_dividend['date'] = df_dividend['date'].astype(str)
        df_dividend['date'] = df_dividend['date'].str.replace('-', '')

        start_date_f = end_date_f
        ten_years_later = datetime.strptime(start_date_f, "%Y-%m-%d") + timedelta(days=3652)
        ten_years_later = ten_years_later.strftime("%Y-%m-%d")
        end_date_f = ten_years_later
        equity_data = pd.concat([equity_data, df_equity], ignore_index=True)
        dividend_data = pd.concat([dividend_data, df_dividend], ignore_index=True)
    equity_data = equity_data.drop_duplicates(subset='date')
    equity_data = equity_data.sort_values(by='date')
    equity_data = equity_data.reset_index(drop=True)
    equity_data['date'] = pd.to_datetime(equity_data['date'], format='%Y%m%d')
    equity_data = equity_data[~equity_data['date'].duplicated(keep='first')]
    equity_data.set_index('date', inplace=True)
    equity_data = equity_data.resample('Q').fillna(method='ffill')
    equity_data.reset_index(inplace=True)
    equity_data['date'] = equity_data['date'].astype(str)
    equity_data['date'] = equity_data['date'].str.replace('-', '')
    equity_data = equity_data.rename(columns={'date': 'endDate'})
    # equity_data.insert(0, 'symbol', symbol_full)
    equity_data = equity_data.ffill()

    dividend_data = dividend_data.drop_duplicates(subset='date')
    dividend_data = dividend_data.sort_values(by='date')
    dividend_data = dividend_data.reset_index(drop=True)
    dividend_data['date'] = pd.to_datetime(dividend_data['date'], format='%Y%m%d')
    dividend_data = dividend_data[~dividend_data['date'].duplicated(keep='first')]
    dividend_data.set_index('date', inplace=True)
    dividend_data = dividend_data.resample('Q').fillna(method='ffill')
    dividend_data.reset_index(inplace=True)
    dividend_data['date'] = dividend_data['date'].astype(str)
    dividend_data['date'] = dividend_data['date'].str.replace('-', '')
    dividend_data = dividend_data.rename(columns={'date': 'endDate'})
    dividend_data.fillna(0, inplace=True)
    dividend_data = dividend_data.merge(equity_data, on='endDate', how='outer')
    dividend_data.fillna(0, inplace=True)
    dividend_data = dividend_data.sort_values(by='endDate', ascending=True)
    dividend_data = dividend_data.reset_index(drop=True)
    dividend_data.insert(0, 'symbol', symbol_full)
    # dividend_data = dividend_data.drop(dividend_data.index[-1])
    return dividend_data

def write_dividend_data_into_table(stock_symbol_data):
    dividend_exchange_name = 'dividend_a'
    #建立连接
    metadata = MetaData()
    dividend_table = Table(dividend_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(dividend_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        ipo_date = str(stock_data['ipoDate'])
        if ipo_date == '':
            ipo_date = '19000101'
        ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
        last_date = conn.execute(f"SELECT MAX(endDate) FROM {dividend_exchange_name} WHERE symbol = '{symbol}'")
        last_date = last_date.scalar()
        if last_date is None:
            last_date = ipo_date
        else:
            last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
        date_format = "%Y-%m-%d"
        last_date = datetime.strptime(last_date, date_format)
        # last_date = last_date + timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        try:
            dividend_data = get_symbol_dividend_data(symbol, last_date)
        except:
            continue
        insert_num = 0
        for index, row in dividend_data.iterrows():
            data = row.to_dict()
            try:
                conn.execute(dividend_table.insert(), data)
                insert_num += 1
            except:
                pass
        print('insert_num:' + str(insert_num))
        
def write_exchange_dividend_data():
    exchange = 'a'
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    dividend_exchange_name = 'dividend_' + exchange
    if dividend_exchange_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        # 如果表格存在，则输出提示信息
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入dividend_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(dividend_exchange_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(dividend_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('bonusSharesFromProfit', String(50)),
                      Column('bonusSharesFromCapitalReserve', String(50)),
                      Column('dividend', String(50)),
                      Column('numberOfShares', String(50)),
                      Column('outstandingSharesA', String(50)),
                      Column('limitedSharesA', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入dividend_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_dividend_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(dividend_exchange_name + ' data inited!')

def get_symbol_income_data(symbol_full, start_date, fs_type):
    symbol = symbol_full.split('.')[0]
    start_date = get_pre_quarter_end_date(start_date)
    start_date = datetime.strptime(start_date, "%Y%m%d")
    start_date = start_date.strftime("%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    ten_years_later = start_date + timedelta(days=3652)
    ten_years_later = ten_years_later.strftime("%Y-%m-%d")
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    start_date_f = start_date.strftime("%Y-%m-%d")
    end_date_f = ten_years_later
    income_data = pd.DataFrame()
    while True:
        if start_date_f >= today:
            break
        # if end_date_f > today:
        #     end_date_f = today
        datas_income = {
            "token": lxr_token,
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCodes": [symbol],
            "metricsList": [
                "q.ps.oi.t",
                "q.ps.oc.t",
                "q.ps.rade.t",
                "q.ps.ae.t",
                "q.ps.se.t",
                "q.ps.fe.t",
                "q.ps.iiife.t",
                "q.ps.ieife.t",
                "q.cfs.dofx_dooaga_dopba.t",
                "q.cfs.daaorei.t",
                "q.cfs.aoia.t",
                "q.cfs.aoltde.t",
                "q.ps.ebitda.t",
                "q.ps.toi.t",
                "q.ps.ivi.t",
                "q.ps.ei.t",
                "q.ps.ebit.t",
                "q.ps.ite.t",
                "q.ps.np.t",
                "q.ps.beps.t",
                "q.ps.deps.t",
                "q.ps.npatoshopc.t",
                "q.bs.sc.t",
            ]
        }
        if fs_type == 'non_financial':
            r_income = requests.post('https://open.lixinger.com/api/cn/company/fs/non_financial', json=datas_income)
        elif fs_type == 'bank':
            r_income = requests.post('https://open.lixinger.com/api/cn/company/fs/bank', json=datas_income)
        elif fs_type == 'security':
            r_income = requests.post('https://open.lixinger.com/api/cn/company/fs/security', json=datas_income)
        elif fs_type == 'insurance':
            r_income = requests.post('https://open.lixinger.com/api/cn/company/fs/insurance', json=datas_income)
        elif fs_type == 'other_financial':
            r_income = requests.post('https://open.lixinger.com/api/cn/company/fs/other_financial', json=datas_income)
        r_income = json_normalize(r_income.json()['data'])
        r_income = r_income.fillna(0)

        try:
            r_income.rename(columns={'date': 'endDate'}, inplace=True)
            r_income['endDate'] = pd.to_datetime(r_income['endDate'])
            r_income['endDate'] = r_income['endDate'].dt.strftime("%Y%m%d")
        except:
            try:
                r_income = r_income.drop('date', axis=1)
            except:
                pass
            r_income['endDate'] = '19000101'
        try:
            r_income = r_income.drop('currency', axis=1)
            r_income = r_income.drop('reportDate', axis=1)
            r_income = r_income.drop('reportType', axis=1)
            r_income = r_income.drop('standardDate', axis=1)
            r_income = r_income.drop('stockCode', axis=1)
        except:
            pass
        if 'q.ps.oi.t' not in r_income.columns:
            r_income['q.ps.oi.t'] = 0
        r_income.rename(columns={'q.ps.oi.t': 'revenue'}, inplace=True)
        if 'q.ps.oc.t' not in r_income.columns:
            r_income['q.ps.oc.t'] = 0
        r_income.rename(columns={'q.ps.oc.t': 'costOfRevenue'}, inplace=True)
        if 'q.ps.rade.t' not in r_income.columns:
            r_income['q.ps.rade.t'] = 0
        r_income.rename(columns={'q.ps.rade.t': 'researchAndDevelopmentExpenses'}, inplace=True)
        if 'q.ps.ae.t' not in r_income.columns:
            r_income['q.ps.ae.t'] = 0
        r_income.rename(columns={'q.ps.ae.t': 'generalAndAdministrativeExpenses'}, inplace=True)
        if 'q.ps.se.t' not in r_income.columns:
            r_income['q.ps.se.t'] = 0
        r_income.rename(columns={'q.ps.se.t': 'sellingAndMarketingExpenses'}, inplace=True)
        if 'q.ps.fe.t' not in r_income.columns:
            r_income['q.ps.fe.t'] = 0
        r_income.rename(columns={'q.ps.fe.t': 'otherExpenses'}, inplace=True)#这里其它费用我们只取财务费用了
        if 'q.ps.iiife.t' not in r_income.columns:
            r_income['q.ps.iiife.t'] = 0
        r_income.rename(columns={'q.ps.iiife.t': 'interestIncome'}, inplace=True)
        if 'q.ps.ieife.t' not in r_income.columns:
            r_income['q.ps.ieife.t'] = 0
        r_income.rename(columns={'q.ps.ieife.t': 'interestExpense'}, inplace=True)
        r_income['interestExpense'] = r_income['interestExpense'].fillna(0)
        r_income['interestIncome'] = r_income['interestIncome'].fillna(0)
        r_income['researchAndDevelopmentExpenses'] = r_income['researchAndDevelopmentExpenses'].fillna(0)
        r_income['grossProfit'] = r_income['revenue'] - r_income['costOfRevenue']
        r_income['grossProfitRatio'] = 0#这里后面合并
        rade = r_income.pop('researchAndDevelopmentExpenses')
        r_income['researchAndDevelopmentExpenses'] = rade
        ae = r_income.pop('generalAndAdministrativeExpenses')
        r_income['generalAndAdministrativeExpenses'] = ae
        se = r_income.pop('sellingAndMarketingExpenses')
        r_income['sellingAndMarketingExpenses'] = se
        r_income['sellingGeneralAndAdministrativeExpenses'] = r_income['generalAndAdministrativeExpenses'] + \
                                                              r_income['sellingAndMarketingExpenses']
        fe = r_income.pop('otherExpenses')
        r_income['otherExpenses'] = fe
        r_income['operatingExpenses'] = r_income['sellingGeneralAndAdministrativeExpenses'] + \
                                        r_income['researchAndDevelopmentExpenses']
        r_income['costAndExpenses'] = r_income['operatingExpenses'] + r_income['otherExpenses'] + \
                                      r_income['costOfRevenue']
        iiife = r_income.pop('interestIncome')
        r_income['interestIncome'] = iiife
        ieife = r_income.pop('interestExpense')
        r_income['interestExpense'] = ieife
        if 'q.cfs.dofx_dooaga_dopba.t' not in r_income.columns:
            r_income['q.cfs.dofx_dooaga_dopba.t'] = 0
        r_income['q.cfs.dofx_dooaga_dopba.t'] = r_income['q.cfs.dofx_dooaga_dopba.t'].fillna(0)
        if 'q.cfs.daaorei.t' not in r_income.columns:
            r_income['q.cfs.daaorei.t'] = 0
        r_income['q.cfs.daaorei.t'] = r_income['q.cfs.daaorei.t'].fillna(0)
        if 'q.cfs.aoia.t' not in r_income.columns:
            r_income['q.cfs.aoia.t'] = 0
        r_income['q.cfs.aoia.t'] = r_income['q.cfs.aoia.t'].fillna(0)
        if 'q.cfs.aoltde.t' not in r_income.columns:
            r_income['q.cfs.aoltde.t'] = 0
        r_income['q.cfs.aoltde.t'] = r_income['q.cfs.aoltde.t'].fillna(0)
        r_income['depreciationAndAmortization'] = r_income['q.cfs.dofx_dooaga_dopba.t'] + r_income['q.cfs.daaorei.t'] + \
                                                  r_income['q.cfs.aoia.t'] + r_income['q.cfs.aoltde.t']
        r_income = r_income.drop('q.cfs.dofx_dooaga_dopba.t', axis=1)
        r_income = r_income.drop('q.cfs.daaorei.t', axis=1)
        r_income = r_income.drop('q.cfs.aoia.t', axis=1)
        r_income = r_income.drop('q.cfs.aoltde.t', axis=1)
        if 'q.ps.ebitda.t' not in r_income.columns:
            r_income['q.ps.ebitda.t'] = 0
        r_income.rename(columns={'q.ps.ebitda.t': 'ebitda'}, inplace=True)
        ebitda = r_income.pop('ebitda')
        r_income['ebitda'] = ebitda
        r_income['ebitda'] = r_income['ebitda'].fillna(0)
        r_income['ebitdaratio'] = 0#合并后再计算
        if 'q.ps.toi.t' not in r_income.columns:
            r_income['q.ps.toi.t'] = 0
        r_income['operatingIncome'] = r_income['q.ps.toi.t'] - r_income['costOfRevenue'] - r_income['operatingExpenses']
        r_income = r_income.drop('q.ps.toi.t', axis=1)
        r_income['operatingIncomeRatio'] = 0#合并后再计算
        if 'q.ps.ivi.t' not in r_income.columns:
            r_income['q.ps.ivi.t'] = 0
        if 'q.ps.ei.t' not in r_income.columns:
            r_income['q.ps.ei.t'] = 0
        r_income['totalOtherIncomeExpensesNet'] = r_income['interestIncome'] + r_income['q.ps.ivi.t'] + r_income['q.ps.ei.t'] - r_income['interestExpense']
        r_income = r_income.drop('q.ps.ivi.t', axis=1)
        r_income = r_income.drop('q.ps.ei.t', axis=1)
        if 'q.ps.ebit.t' not in r_income.columns:
            r_income['q.ps.ebit.t'] = 0
        r_income.rename(columns={'q.ps.ebit.t': 'incomeBeforeTax'}, inplace=True)
        income_before_tax = r_income.pop('incomeBeforeTax')
        r_income['incomeBeforeTax'] = income_before_tax
        r_income['incomeBeforeTax'] = r_income['incomeBeforeTax'].fillna(0)
        r_income['incomeBeforeTaxRatio'] = 0
        if 'q.ps.ite.t' not in r_income.columns:
            r_income['q.ps.ite.t'] = 0
        r_income.rename(columns={'q.ps.ite.t': 'incomeTaxExpense'}, inplace=True)
        income_tax_expense = r_income.pop('incomeTaxExpense')
        r_income['incomeTaxExpense'] = income_tax_expense
        if 'q.ps.np.t' not in r_income.columns:
            r_income['q.ps.np.t'] = 0
        r_income.rename(columns={'q.ps.np.t': 'netIncome'}, inplace=True)
        net_income = r_income.pop('netIncome')
        r_income['netIncome'] = net_income
        r_income['netIncomeRatio'] = 0
        if 'q.ps.beps.t' not in r_income.columns:
            r_income['q.ps.beps.t'] = 0
        r_income.rename(columns={'q.ps.beps.t': 'eps'}, inplace=True)
        eps = r_income.pop('eps')
        r_income['eps'] = eps
        if 'q.ps.deps.t' not in r_income.columns:
            r_income['q.ps.deps.t'] = 0
        r_income.rename(columns={'q.ps.deps.t': 'epsdiluted'}, inplace=True)
        epsdiluted = r_income.pop('epsdiluted')
        r_income['epsdiluted'] = epsdiluted
        if 'q.bs.sc.t' not in r_income.columns:
            r_income['q.bs.sc.t'] = 0
        r_income['weightedAverageShsOut'] = r_income['q.bs.sc.t']
        if 'q.ps.npatoshopc.t' not in r_income.columns:
            r_income['q.ps.npatoshopc.t'] = 0
        r_income['weightedAverageShsOutDil'] = r_income['q.ps.npatoshopc.t']/(r_income['epsdiluted'] + 1)
        r_income = r_income.drop('q.ps.npatoshopc.t', axis=1)
        r_income = r_income.drop('q.bs.sc.t', axis=1)
        r_income.insert(0, 'symbol', symbol_full)
        for i in range(len(r_income) - 1):
            if '0331' not in str(r_income.iloc[i]['endDate']):
                r_income.loc[i, 'revenue'] = r_income.loc[i, 'revenue'] - r_income.loc[i+1, 'revenue']
                r_income.loc[i, 'costOfRevenue'] = r_income.loc[i, 'costOfRevenue'] - r_income.loc[i+1, 'costOfRevenue']
                r_income.loc[i, 'grossProfit'] = r_income.loc[i, 'grossProfit'] - r_income.loc[i+1, 'grossProfit']
                r_income.loc[i, 'researchAndDevelopmentExpenses'] = r_income.loc[i, 'researchAndDevelopmentExpenses'] - r_income.loc[i+1, 'researchAndDevelopmentExpenses']
                r_income.loc[i, 'generalAndAdministrativeExpenses'] = r_income.loc[i, 'generalAndAdministrativeExpenses'] - r_income.loc[i+1, 'generalAndAdministrativeExpenses']
                r_income.loc[i, 'sellingAndMarketingExpenses'] = r_income.loc[i, 'sellingAndMarketingExpenses'] - r_income.loc[i+1, 'sellingAndMarketingExpenses']
                r_income.loc[i, 'sellingGeneralAndAdministrativeExpenses'] = r_income.loc[i, 'sellingGeneralAndAdministrativeExpenses'] - r_income.loc[i+1, 'sellingGeneralAndAdministrativeExpenses']
                r_income.loc[i, 'otherExpenses'] = r_income.loc[i, 'otherExpenses'] - r_income.loc[i+1, 'otherExpenses']
                r_income.loc[i, 'operatingExpenses'] = r_income.loc[i, 'operatingExpenses'] - r_income.loc[i+1, 'operatingExpenses']
                r_income.loc[i, 'costAndExpenses'] = r_income.loc[i, 'costAndExpenses'] - r_income.loc[i+1, 'costAndExpenses']
                r_income.loc[i, 'interestIncome'] = r_income.loc[i, 'interestIncome'] - r_income.loc[i+1, 'interestIncome']
                r_income.loc[i, 'interestExpense'] = r_income.loc[i, 'interestExpense'] - r_income.loc[i+1, 'interestExpense']
                r_income.loc[i, 'depreciationAndAmortization'] = r_income.loc[i, 'depreciationAndAmortization'] - r_income.loc[i+1, 'depreciationAndAmortization']
                r_income.loc[i, 'ebitda'] = r_income.loc[i, 'ebitda'] - r_income.loc[i+1, 'ebitda']
                r_income.loc[i, 'operatingIncome'] = r_income.loc[i, 'operatingIncome'] - r_income.loc[i+1, 'operatingIncome']
                r_income.loc[i, 'totalOtherIncomeExpensesNet'] = r_income.loc[i, 'totalOtherIncomeExpensesNet'] - r_income.loc[i+1, 'totalOtherIncomeExpensesNet']
                r_income.loc[i, 'incomeBeforeTax'] = r_income.loc[i, 'incomeBeforeTax'] - r_income.loc[i+1, 'incomeBeforeTax']
                r_income.loc[i, 'incomeTaxExpense'] = r_income.loc[i, 'incomeTaxExpense'] - r_income.loc[i+1, 'incomeTaxExpense']
                r_income.loc[i, 'netIncome'] = r_income.loc[i, 'netIncome'] - r_income.loc[i+1, 'netIncome']
                r_income.loc[i, 'eps'] = r_income.loc[i, 'eps'] - r_income.loc[i+1, 'eps']
                r_income.loc[i, 'epsdiluted'] = r_income.loc[i, 'epsdiluted'] - r_income.loc[i+1, 'epsdiluted']
        r_income['grossProfitRatio'] = r_income['grossProfit']/(r_income['revenue'] + 1)
        r_income['ebitdaratio'] = r_income['ebitda']/(r_income['revenue'] + 1)
        r_income['operatingIncomeRatio'] = r_income['operatingIncome']/(r_income['revenue'] + 1)
        r_income['incomeBeforeTaxRatio'] = r_income['incomeBeforeTax']/(r_income['revenue'] + 1)
        r_income['netIncomeRatio'] = r_income['netIncome']/(r_income['revenue'] + 1)
        try:
            r_income = r_income.drop(r_income.index[-1])
        except:
            pass
        start_date_f = end_date_f
        start_date_f = get_pre_quarter_end_date(start_date_f)
        start_date_f = datetime.strptime(start_date_f, "%Y%m%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")
        start_date_f = datetime.strptime(start_date_f, "%Y-%m-%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")

        ten_years_later = datetime.strptime(start_date_f, "%Y-%m-%d") + timedelta(days=3652)
        ten_years_later = ten_years_later.strftime("%Y-%m-%d")
        end_date_f = ten_years_later
        income_data = pd.concat([income_data, r_income], ignore_index=True)
        # print(income_data)
    income_data = income_data.drop_duplicates(subset='endDate')
    income_data = income_data.sort_values(by='endDate', ascending=False)
    income_data = income_data.reset_index(drop=True)
    income_data = income_data.fillna(0)
    # print(income_data)
    return income_data

def write_income_data_into_table(stock_symbol_data):
    income_exchange_name = 'income_a'
    #建立连接
    metadata = MetaData()
    income_table = Table(income_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    # 获取表的列顺序
    inspector = inspect(engine)
    columns = inspector.get_columns(income_exchange_name)
    column_names = [column['name'] for column in columns]
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(income_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        ipo_date = str(stock_data['ipoDate'])
        fs_type = stock_data['fsType']
        if ipo_date == '':
            ipo_date = '19000101'
        ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
        last_date = conn.execute(f"SELECT MAX(endDate) FROM {income_exchange_name} WHERE symbol = '{symbol}'")
        last_date = last_date.scalar()
        if last_date is None:
            last_date = ipo_date
        else:
            last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
        date_format = "%Y-%m-%d"
        last_date = datetime.strptime(last_date, date_format)
        # last_date = last_date + timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        if fs_type != 'non_financial':
            continue
        income_data = get_symbol_income_data(symbol, last_date, fs_type)
        income_data = income_data[column_names]
        insert_num = 0
        for index, row in income_data.iterrows():
            data = row.to_dict()
            try:
                conn.execute(income_table.insert(), data)
                insert_num += 1
            except:
                pass
        print('insert_num:' + str(insert_num))

def write_exchange_income_data():
    exchange = 'a'
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    income_exchange_name = 'income_' + exchange
    if income_exchange_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        # 如果表格存在，则输出提示信息
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入income_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(income_exchange_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(income_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('revenue', String(50)),
                      Column('costOfRevenue', String(50)),
                      Column('grossProfit', String(50)),
                      Column('grossProfitRatio', String(50)),
                      Column('researchAndDevelopmentExpenses', String(50)),
                      Column('generalAndAdministrativeExpenses', String(50)),
                      Column('sellingAndMarketingExpenses', String(50)),
                      Column('sellingGeneralAndAdministrativeExpenses', String(50)),
                      Column('otherExpenses', String(50)),
                      Column('operatingExpenses', String(50)),
                      Column('costAndExpenses', String(50)),
                      Column('interestIncome', String(50)),
                      Column('interestExpense', String(50)),
                      Column('depreciationAndAmortization', String(50)),
                      Column('ebitda', String(50)),
                      Column('ebitdaratio', String(50)),
                      Column('operatingIncome', String(50)),
                      Column('operatingIncomeRatio', String(50)),
                      Column('totalOtherIncomeExpensesNet', String(50)),
                      Column('incomeBeforeTax', String(50)),
                      Column('incomeBeforeTaxRatio', String(50)),
                      Column('incomeTaxExpense', String(50)),
                      Column('netIncome', String(50)),
                      Column('netIncomeRatio', String(50)),
                      Column('eps', String(50)),
                      Column('epsdiluted', String(50)),
                      Column('weightedAverageShsOut', String(50)),
                      Column('weightedAverageShsOutDil', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入income_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_income_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(income_exchange_name + ' data inited!')

def get_symbol_balance_data(symbol_full, start_date, fs_type):
    symbol = symbol_full.split('.')[0]
    start_date = get_pre_quarter_end_date(start_date)
    start_date = datetime.strptime(start_date, "%Y%m%d")
    start_date = start_date.strftime("%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    ten_years_later = start_date + timedelta(days=3652)
    ten_years_later = ten_years_later.strftime("%Y-%m-%d")
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    start_date_f = start_date.strftime("%Y-%m-%d")
    end_date_f = ten_years_later
    balance_data = pd.DataFrame()
    while True:
        if start_date_f >= today:
            break
        # if end_date_f > today:
        #     end_date_f = today
        datas_balance = {
            "token": lxr_token,
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCodes": [symbol],
            "metricsList": [
                "q.bs.cabb.t",
                "q.bs.tfa.t",
                "q.bs.ar.t",
                "q.bs.i.t",
                "q.bs.ats.t",
                "q.bs.oca.t",
                "q.bs.ocri.t",
                "q.bs.fa.t",
                "q.bs.dofa.t",
                "q.bs.gw.t",
                "q.bs.ia.t",
                "q.bs.cri.t",
                "q.bs.ocri.t",
                "q.bs.ltei.t",
                "q.bs.oeii.t",
                "q.bs.rei.t",
                "q.bs.dita.t",
                "q.bs.onca.t",
                "q.bs.tnca.t",
                "q.bs.ap.t",
                "q.bs.cal.t",
                "q.bs.tp.t",
                "q.bs.afc.t",
                "q.bs.ocl.t",
                "q.bs.tcl.t",
                "q.bs.ltl.t",
                "q.bs.bp.t",
                "q.bs.ltdi.t",
                "q.bs.ditl.t",
                "q.bs.oncl.t",
                "q.bs.tncl.t",
                "q.bs.ll.t",
                "q.bs.psioei.t",
                "q.bs.sc.t",
                "q.bs.rtp.t",
                "q.bs.oci.t",
                "q.bs.oei.t",
                "q.bs.etmsh.t",
                "q.bs.lwi.t",
            ]
        }
        if fs_type == 'non_financial':
            r_balance = requests.post('https://open.lixinger.com/api/cn/company/fs/non_financial', json=datas_balance)
        elif fs_type == 'bank':
            r_balance = requests.post('https://open.lixinger.com/api/cn/company/fs/bank', json=datas_balance)
        elif fs_type == 'security':
            r_balance = requests.post('https://open.lixinger.com/api/cn/company/fs/security', json=datas_balance)
        elif fs_type == 'insurance':
            r_balance = requests.post('https://open.lixinger.com/api/cn/company/fs/insurance', json=datas_balance)
        elif fs_type == 'other_financial':
            r_balance = requests.post('https://open.lixinger.com/api/cn/company/fs/other_financial', json=datas_balance)
        r_balance = json_normalize(r_balance.json()['data'])
        r_balance = r_balance.fillna(0)
        try:
            r_balance.rename(columns={'date': 'endDate'}, inplace=True)
            r_balance['endDate'] = pd.to_datetime(r_balance['endDate'])
            r_balance['endDate'] = r_balance['endDate'].dt.strftime("%Y%m%d")
        except:
            try:
                r_balance = r_balance.drop('date', axis=1)
            except:
                pass
            r_balance['endDate'] = '19000101'
        try:
            r_balance = r_balance.drop('currency', axis=1)
            r_balance = r_balance.drop('reportDate', axis=1)
            r_balance = r_balance.drop('reportType', axis=1)
            r_balance = r_balance.drop('standardDate', axis=1)
            r_balance = r_balance.drop('stockCode', axis=1)
        except:
            pass
        if 'q.bs.cabb.t' not in r_balance.columns:
            r_balance['q.bs.cabb.t'] = 0
        r_balance.rename(columns={'q.bs.cabb.t': 'cashAndCashEquivalents'}, inplace=True)
        if 'q.bs.tfa.t' not in r_balance.columns:
            r_balance['q.bs.tfa.t'] = 0
        r_balance.rename(columns={'q.bs.tfa.t': 'shortTermInvestments'}, inplace=True)
        r_balance['cashAndShortTermInvestments'] = r_balance['cashAndCashEquivalents'] + \
                                                   r_balance['shortTermInvestments']
        if 'q.bs.ar.t' not in r_balance.columns:
            r_balance['q.bs.ar.t'] = 0
        r_balance.rename(columns={'q.bs.ar.t': 'netReceivables'}, inplace=True)
        if 'q.bs.i.t' not in r_balance.columns:
            r_balance['q.bs.i.t'] = 0
        r_balance.rename(columns={'q.bs.i.t': 'inventory'}, inplace=True)
        if 'q.bs.ats.t' not in r_balance.columns:
            r_balance['q.bs.ats.t'] = 0
        if 'q.bs.oca.t' not in r_balance.columns:
            r_balance['q.bs.oca.t'] = 0
        if 'q.bs.ocri.t' not in r_balance.columns:
            r_balance['q.bs.ocri.t'] = 0
        r_balance['otherCurrentAssets'] = r_balance['q.bs.ats.t'] + r_balance['q.bs.oca.t'] + r_balance['q.bs.ocri.t']
        r_balance['totalCurrentAssets'] = r_balance['cashAndCashEquivalents'] + r_balance['netReceivables'] + r_balance['inventory'] + r_balance['q.bs.ats.t'] + r_balance['q.bs.oca.t']
        r_balance = r_balance.drop('q.bs.ats.t', axis=1)
        r_balance = r_balance.drop('q.bs.oca.t', axis=1)
        r_balance = r_balance.drop('q.bs.ocri.t', axis=1)
        if 'q.bs.fa.t' not in r_balance.columns:
            r_balance['q.bs.fa.t'] = 0
        if 'q.bs.dofa.t' not in r_balance.columns:
            r_balance['q.bs.dofa.t'] = 0
        r_balance['propertyPlantEquipmentNet'] = r_balance['q.bs.fa.t'] - r_balance['q.bs.dofa.t']
        r_balance = r_balance.drop('q.bs.fa.t', axis=1)
        r_balance = r_balance.drop('q.bs.dofa.t', axis=1)
        if 'q.bs.gw.t' not in r_balance.columns:
            r_balance['q.bs.gw.t'] = 0
        r_balance.rename(columns={'q.bs.gw.t': 'goodwill'}, inplace=True)
        if 'q.bs.ia.t' not in r_balance.columns:
            r_balance['q.bs.ia.t'] = 0
        r_balance.rename(columns={'q.bs.ia.t': 'intangibleAssets'}, inplace=True)
        r_balance['goodwillAndIntangibleAssets'] = r_balance['goodwill'] + r_balance['intangibleAssets']
        if 'q.bs.cri.t' not in r_balance.columns:
            r_balance['q.bs.cri.t'] = 0
        if 'q.bs.ocri.t' not in r_balance.columns:
            r_balance['q.bs.ocri.t'] = 0
        if 'q.bs.ltei.t' not in r_balance.columns:
            r_balance['q.bs.ltei.t'] = 0
        if 'q.bs.bs.oeii.t' not in r_balance.columns:
            r_balance['q.bs.oeii.t'] = 0
        if 'q.bs.bs.rei.t' not in r_balance.columns:
            r_balance['q.bs.rei.t'] = 0
        r_balance['longTermInvestments'] = r_balance['q.bs.cri.t'] + r_balance['q.bs.ocri.t'] + r_balance['q.bs.ltei.t'] + r_balance['q.bs.oeii.t'] + r_balance['q.bs.rei.t']
        r_balance = r_balance.drop('q.bs.cri.t', axis=1)
        r_balance = r_balance.drop('q.bs.ocri.t', axis=1)
        r_balance = r_balance.drop('q.bs.ltei.t', axis=1)
        r_balance = r_balance.drop('q.bs.oeii.t', axis=1)
        r_balance = r_balance.drop('q.bs.rei.t', axis=1)
        if 'q.bs.dita.t' not in r_balance.columns:
            r_balance['q.bs.dita.t'] = 0
        r_balance.rename(columns={'q.bs.dita.t': 'taxAssets'}, inplace=True)
        if 'q.bs.onca.t' not in r_balance.columns:
            r_balance['q.bs.onca.t'] = 0
        r_balance.rename(columns={'q.bs.onca.t': 'otherNonCurrentAssets'}, inplace=True)
        if 'q.bs.tnca.t' not in r_balance.columns:
            r_balance['q.bs.tnca.t'] = 0
        r_balance.rename(columns={'q.bs.tnca.t': 'totalNonCurrentAssets'}, inplace=True)
        r_balance['otherAssets'] = r_balance['taxAssets'] + r_balance['otherNonCurrentAssets']
        r_balance['totalAssets'] = r_balance['totalCurrentAssets'] + r_balance['totalNonCurrentAssets']
        r_balance.rename(columns={'q.bs.ap.t': 'accountPayables'}, inplace=True)
        if 'q.bs.cal.t' not in r_balance.columns:
            r_balance['q.bs.cal.t'] = 0
        r_balance.rename(columns={'q.bs.cal.t': 'shortTermDebt'}, inplace=True)
        if 'q.bs.tp.t' not in r_balance.columns:
            r_balance['q.bs.tp.t'] = 0
        r_balance.rename(columns={'q.bs.tp.t': 'taxPayables'}, inplace=True)
        if 'q.bs.afc.t' not in r_balance.columns:
            r_balance['q.bs.afc.t'] = 0
        r_balance.rename(columns={'q.bs.afc.t': 'deferredRevenue'}, inplace=True)
        if 'q.bs.ocl.t' not in r_balance.columns:
            r_balance['q.bs.ocl.t'] = 0
        r_balance.rename(columns={'q.bs.ocl.t': 'otherCurrentLiabilities'}, inplace=True)
        if 'q.bs.tcl.t' not in r_balance.columns:
            r_balance['q.bs.tcl.t'] = 0
        r_balance.rename(columns={'q.bs.tcl.t': 'totalCurrentLiabilities'}, inplace=True)
        if 'q.bs.ltl.t' not in r_balance.columns:
            r_balance['q.bs.ltl.t'] = 0
        if 'q.bs.bp.t' not in r_balance.columns:
            r_balance['q.bs.bp.t'] = 0
        r_balance['longTermDebt'] = r_balance['q.bs.ltl.t'] + r_balance['q.bs.bp.t']
        r_balance = r_balance.drop('q.bs.ltl.t', axis=1)
        r_balance = r_balance.drop('q.bs.bp.t', axis=1)
        r_balance.rename(columns={'q.bs.ltdi.t': 'deferredRevenueNonCurrent'}, inplace=True)
        r_balance.rename(columns={'q.bs.ditl.t': 'deferredTaxLiabilitiesNonCurrent'}, inplace=True)
        if 'q.bs.oncl.t' not in r_balance.columns:
            r_balance['q.bs.oncl.t'] = 0
        r_balance.rename(columns={'q.bs.oncl.t': 'otherNonCurrentLiabilities'}, inplace=True)
        if 'q.bs.tncl.t' not in r_balance.columns:
            r_balance['q.bs.tncl.t'] = 0
        r_balance.rename(columns={'q.bs.tncl.t': 'totalNonCurrentLiabilities'}, inplace=True)
        r_balance['otherLiabilities'] = r_balance['otherCurrentLiabilities'] + r_balance['otherNonCurrentLiabilities']
        r_balance.rename(columns={'q.bs.ll.t': 'capitalLeaseObligations'}, inplace=True)
        r_balance['totalLiabilities'] = r_balance['totalCurrentLiabilities'] + r_balance['totalNonCurrentLiabilities']
        r_balance.rename(columns={'q.bs.psioei.t': 'preferredStock'}, inplace=True)
        r_balance.rename(columns={'q.bs.sc.t': 'commonStock'}, inplace=True)
        r_balance.rename(columns={'q.bs.rtp.t': 'retainedEarnings'}, inplace=True)
        r_balance.rename(columns={'q.bs.oci.t': 'accumulatedOtherComprehensiveIncomeLoss'}, inplace=True)
        if 'q.bs.oei.t' not in r_balance.columns:
            r_balance['q.bs.oei.t'] = 0
        r_balance.rename(columns={'q.bs.oei.t': 'othertotalStockholdersEquity'}, inplace=True)
        r_balance['totalStockholdersEquity'] = r_balance['totalAssets'] - r_balance['totalLiabilities'] - r_balance['othertotalStockholdersEquity']
        r_balance['totalEquity'] = r_balance['totalAssets'] - r_balance['totalLiabilities']
        r_balance['totalLiabilitiesAndStockholdersEquity'] = r_balance['totalStockholdersEquity'] + r_balance['totalLiabilities']
        r_balance.rename(columns={'q.bs.etmsh.t': 'minorityInterest'}, inplace=True)
        r_balance['totalLiabilitiesAndTotalEquity'] = r_balance['totalEquity'] + r_balance['totalLiabilities']
        r_balance['totalInvestments'] = r_balance['shortTermInvestments'] + r_balance['longTermInvestments']
        r_balance['totalDebt'] = r_balance['shortTermDebt'] + r_balance['longTermDebt']
        r_balance.rename(columns={'q.bs.lwi.t': 'netDebt'}, inplace=True)
        r_balance.insert(0, 'symbol', symbol_full)

        try:
            r_balance = r_balance.drop(r_balance.index[-1])
        except:
            pass
        start_date_f = end_date_f
        start_date_f = get_pre_quarter_end_date(start_date_f)
        start_date_f = datetime.strptime(start_date_f, "%Y%m%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")
        start_date_f = datetime.strptime(start_date_f, "%Y-%m-%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")

        ten_years_later = datetime.strptime(start_date_f, "%Y-%m-%d") + timedelta(days=3652)
        ten_years_later = ten_years_later.strftime("%Y-%m-%d")
        end_date_f = ten_years_later
        balance_data = pd.concat([balance_data, r_balance], ignore_index=True)
        # print(balance_data)
    balance_data = balance_data.drop_duplicates(subset='endDate')

    balance_data = balance_data.sort_values(by='endDate', ascending=False)
    balance_data = balance_data.reset_index(drop=True)
    balance_data = balance_data.fillna(0)
    # print(balance_data)
    return balance_data

def write_balance_data_into_table(stock_symbol_data):
    balance_exchange_name = 'balance_a'
    #建立连接
    metadata = MetaData()
    balance_table = Table(balance_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    # 获取表的列顺序
    inspector = inspect(engine)
    columns = inspector.get_columns(balance_exchange_name)
    column_names = [column['name'] for column in columns]
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(balance_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        ipo_date = str(stock_data['ipoDate'])
        fs_type = stock_data['fsType']
        if ipo_date == '':
            ipo_date = '19000101'
        ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
        last_date = conn.execute(f"SELECT MAX(endDate) FROM {balance_exchange_name} WHERE symbol = '{symbol}'")
        last_date = last_date.scalar()
        if last_date is None:
            last_date = ipo_date
        else:
            last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
        date_format = "%Y-%m-%d"
        last_date = datetime.strptime(last_date, date_format)
        # last_date = last_date + timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        if fs_type != 'non_financial':
            continue
        balance_data = get_symbol_balance_data(symbol, last_date, fs_type)
        balance_data = balance_data[column_names]
        insert_num = 0
        for index, row in balance_data.iterrows():
            data = row.to_dict()
            try:
                conn.execute(balance_table.insert(), data)
                insert_num += 1
            except:
                pass
        print('insert_num:' + str(insert_num))

def write_exchange_balance_data():
    exchange = 'a'
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    balance_exchange_name = 'balance_' + exchange
    if balance_exchange_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        # 如果表格存在，则输出提示信息
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入balance_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(balance_exchange_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(balance_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('cashAndCashEquivalents', String(50)),
                      Column('shortTermInvestments', String(50)),
                      Column('cashAndShortTermInvestments', String(50)),
                      Column('netReceivables', String(50)),
                      Column('inventory', String(50)),
                      Column('otherCurrentAssets', String(50)),
                      Column('totalCurrentAssets', String(50)),
                      Column('propertyPlantEquipmentNet', String(50)),
                      Column('goodwill', String(50)),
                      Column('intangibleAssets', String(50)),
                      Column('goodwillAndIntangibleAssets', String(50)),
                      Column('longTermInvestments', String(50)),
                      Column('taxAssets', String(50)),
                      Column('otherNonCurrentAssets', String(50)),
                      Column('totalNonCurrentAssets', String(50)),
                      Column('otherAssets', String(50)),
                      Column('totalAssets', String(50)),
                      Column('accountPayables', String(50)),
                      Column('shortTermDebt', String(50)),
                      Column('taxPayables', String(50)),
                      Column('deferredRevenue', String(50)),
                      Column('otherCurrentLiabilities', String(50)),
                      Column('totalCurrentLiabilities', String(50)),
                      Column('longTermDebt', String(50)),
                      Column('deferredRevenueNonCurrent', String(50)),
                      Column('deferredTaxLiabilitiesNonCurrent', String(50)),
                      Column('otherNonCurrentLiabilities', String(50)),
                      Column('totalNonCurrentLiabilities', String(50)),
                      Column('otherLiabilities', String(50)),
                      Column('capitalLeaseObligations', String(50)),
                      Column('totalLiabilities', String(50)),
                      Column('preferredStock', String(50)),
                      Column('commonStock', String(50)),
                      Column('retainedEarnings', String(50)),
                      Column('accumulatedOtherComprehensiveIncomeLoss', String(50)),
                      Column('othertotalStockholdersEquity', String(50)),
                      Column('totalStockholdersEquity', String(50)),
                      Column('totalEquity', String(50)),
                      Column('totalLiabilitiesAndStockholdersEquity', String(50)),
                      Column('minorityInterest', String(50)),
                      Column('totalLiabilitiesAndTotalEquity', String(50)),
                      Column('totalInvestments', String(50)),
                      Column('totalDebt', String(50)),
                      Column('netDebt', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入balance_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_balance_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(balance_exchange_name + ' data inited!')

def get_symbol_cashflow_data(symbol_full, start_date, fs_type):
    symbol = symbol_full.split('.')[0]
    start_date = get_pre_quarter_end_date(start_date)
    start_date = datetime.strptime(start_date, "%Y%m%d")
    start_date = start_date.strftime("%Y-%m-%d")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    ten_years_later = start_date + timedelta(days=3652)
    ten_years_later = ten_years_later.strftime("%Y-%m-%d")
    today = datetime.today()
    today = today.strftime("%Y-%m-%d")
    start_date_f = start_date.strftime("%Y-%m-%d")
    end_date_f = ten_years_later
    cashflow_data = pd.DataFrame()
    while True:
        if start_date_f >= today:
            break
        # if end_date_f > today:
        #     end_date_f = today
        datas_cashflow = {
            "token": lxr_token,
            "startDate": start_date_f,
            "endDate": end_date_f,
            "stockCodes": [symbol],
            "metricsList": [
                "q.bs.ditl.t",
                "q.bs.sawp.t",
                "q.cfs.ncffoa.t",
                "q.bs.ar.t",
                "q.bs.ap.t",
                "q.cfs.crrtooa.t",
                "q.cfs.dofx_dooaga_dopba.t",
                "q.cfs.daaorei.t",
                "q.cfs.aoia.t",
                "q.cfs.aoltde.t",
                "q.cfs.cpfpfiaolta.t",
                "q.cfs.ncfffa.t",
                "q.cfs.stcoffia.t",
                "q.cfs.crfii.t",
                "q.cfs.crrtoia.t",
                "q.cfs.cpfbrp.t",
                "q.cfs.cpfdapdoi.t",
                "q.cfs.crfai.t",
                "q.cfs.cprtofa.t",
                "q.cfs.crrtofa.t",
                "q.cfs.iocacedtfier.t",
                "q.m.ncffoaiafa.t",
                "q.cfs.bocaceatpe.t",
                "q.cfs.bocaceatpb.t",
                "q.m.fcf.t",
            ]
        }
        if fs_type == 'non_financial':
            r_cashflow = requests.post('https://open.lixinger.com/api/cn/company/fs/non_financial', json=datas_cashflow)
        elif fs_type == 'bank':
            r_cashflow = requests.post('https://open.lixinger.com/api/cn/company/fs/bank', json=datas_cashflow)
        elif fs_type == 'security':
            r_cashflow = requests.post('https://open.lixinger.com/api/cn/company/fs/security', json=datas_cashflow)
        elif fs_type == 'insurance':
            r_cashflow = requests.post('https://open.lixinger.com/api/cn/company/fs/insurance', json=datas_cashflow)
        elif fs_type == 'other_financial':
            r_cashflow = requests.post('https://open.lixinger.com/api/cn/company/fs/other_financial', json=datas_cashflow)
        r_cashflow = json_normalize(r_cashflow.json()['data'])
        r_cashflow = r_cashflow.fillna(0)
        try:
            r_cashflow.rename(columns={'date': 'endDate'}, inplace=True)
            r_cashflow['endDate'] = pd.to_datetime(r_cashflow['endDate'])
            r_cashflow['endDate'] = r_cashflow['endDate'].dt.strftime("%Y%m%d")
        except:
            try:
                r_cashflow = r_cashflow.drop('date', axis=1)
            except:
                pass
            r_cashflow['endDate'] = '19000101'
        try:
            r_cashflow = r_cashflow.drop('currency', axis=1)
            r_cashflow = r_cashflow.drop('reportDate', axis=1)
            r_cashflow = r_cashflow.drop('reportType', axis=1)
            r_cashflow = r_cashflow.drop('standardDate', axis=1)
            r_cashflow = r_cashflow.drop('stockCode', axis=1)
        except:
            pass
        r_cashflow.rename(columns={'q.bs.ditl.t': 'deferredIncomeTax'}, inplace=True)
        r_cashflow.rename(columns={'q.bs.sawp.t': 'stockBasedCompensation'}, inplace=True)#这里用应付薪酬替代
        if 'q.cfs.ncffoa.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.ncffoa.t'] = 0
        r_cashflow.rename(columns={'q.cfs.ncffoa.t': 'changeInWorkingCapital'}, inplace=True)#这里用经营活动产生的现金流量净额替代
        r_cashflow.rename(columns={'q.bs.ar.t': 'accountsReceivables'}, inplace=True)
        r_cashflow.rename(columns={'q.bs.ap.t': 'accountsPayables'}, inplace=True)
        r_cashflow.rename(columns={'q.cfs.crrtooa.t': 'otherWorkingCapital'}, inplace=True)#这里用收到的其他与经营活动有关现金替代

        if 'q.cfs.dofx_dooaga_dopba.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.dofx_dooaga_dopba.t'] = 0
        if 'q.cfs.daaorei.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.daaorei.t'] = 0
        if 'q.cfs.aoia.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.aoia.t'] = 0
        if 'q.cfs.aoltde.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.aoltde.t'] = 0

        r_cashflow['otherNonCashItems'] = r_cashflow['q.cfs.dofx_dooaga_dopba.t'] + r_cashflow['q.cfs.daaorei.t'] + r_cashflow['q.cfs.aoia.t'] + r_cashflow['q.cfs.aoltde.t']
        r_cashflow = r_cashflow.drop('q.cfs.dofx_dooaga_dopba.t', axis=1)
        r_cashflow = r_cashflow.drop('q.cfs.daaorei.t', axis=1)
        r_cashflow = r_cashflow.drop('q.cfs.aoia.t', axis=1)
        r_cashflow = r_cashflow.drop('q.cfs.aoltde.t', axis=1)
        r_cashflow['netCashProvidedByOperatingActivities'] = r_cashflow['changeInWorkingCapital']
        if 'q.cfs.cpfpfiaolta.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.cpfpfiaolta.t'] = 0
        r_cashflow.rename(columns={'q.cfs.cpfpfiaolta.t': 'investmentsInPropertyPlantAndEquipment'}, inplace=True)
        if 'q.cfs.ncfffa.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.ncfffa.t'] = 0
        r_cashflow.rename(columns={'q.cfs.ncfffa.t': 'acquisitionsNet'}, inplace=True)
        if 'q.cfs.stcoffia.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.stcoffia.t'] = 0
        r_cashflow.rename(columns={'q.cfs.stcoffia.t': 'purchasesOfInvestments'}, inplace=True)
        if 'q.cfs.crfii.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.crfii.t'] = 0
        r_cashflow.rename(columns={'q.cfs.crfii.t': 'salesMaturitiesOfInvestments'}, inplace=True)
        if 'q.cfs.crrtoia.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.crrtoia.t'] = 0
        r_cashflow.rename(columns={'q.cfs.crrtoia.t': 'otherInvestingActivites'}, inplace=True)
        r_cashflow['netCashUsedForInvestingActivites'] = r_cashflow['purchasesOfInvestments']

        if 'q.cfs.cpfbrp.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.cpfbrp.t'] = 0
        if 'q.cfs.cpfdapdoi.t' not in r_cashflow.columns:
            r_cashflow['q.cfs.cpfdapdoi.t'] = 0
        r_cashflow['debtRepayment'] = r_cashflow['q.cfs.cpfbrp.t'] + r_cashflow['q.cfs.cpfdapdoi.t']
        r_cashflow.rename(columns={'q.cfs.crfai.t': 'commonStockIssued'}, inplace=True)
        r_cashflow.rename(columns={'q.cfs.cprtofa.t': 'commonStockRepurchased'}, inplace=True)
        r_cashflow['dividendsPaid'] = r_cashflow['q.cfs.cpfdapdoi.t']
        r_cashflow = r_cashflow.drop('q.cfs.cpfbrp.t', axis=1)
        r_cashflow = r_cashflow.drop('q.cfs.cpfdapdoi.t', axis=1)
        r_cashflow.rename(columns={'q.cfs.crrtofa.t': 'otherFinancingActivites'}, inplace=True)
        r_cashflow['netCashUsedProvidedByFinancingActivities'] = r_cashflow['acquisitionsNet']
        r_cashflow.rename(columns={'q.cfs.iocacedtfier.t': 'effectOfForexChangesOnCash'}, inplace=True)
        r_cashflow.rename(columns={'q.m.ncffoaiafa.t': 'netChangeInCash'}, inplace=True)
        r_cashflow.rename(columns={'q.cfs.bocaceatpe.t': 'cashAtEndOfPeriod'}, inplace=True)
        r_cashflow.rename(columns={'q.cfs.bocaceatpb.t': 'cashAtBeginningOfPeriod'}, inplace=True)
        r_cashflow['operatingCashFlow'] = r_cashflow['changeInWorkingCapital']
        r_cashflow['capitalExpenditure'] = r_cashflow['investmentsInPropertyPlantAndEquipment']
        r_cashflow.rename(columns={'q.m.fcf.t': 'freeCashFlow'}, inplace=True)
        r_cashflow.insert(0, 'symbol', symbol_full)
        for i in range(len(r_cashflow) - 1):
            if '0331' not in str(r_cashflow.iloc[i]['endDate']):
                r_cashflow.loc[i, 'stockBasedCompensation'] = r_cashflow.loc[i, 'stockBasedCompensation'] - r_cashflow.loc[i+1, 'stockBasedCompensation']
                r_cashflow.loc[i, 'otherNonCashItems'] = r_cashflow.loc[i, 'otherNonCashItems'] - r_cashflow.loc[i+1, 'otherNonCashItems']
                r_cashflow.loc[i, 'netCashProvidedByOperatingActivities'] = r_cashflow.loc[i, 'netCashProvidedByOperatingActivities'] - r_cashflow.loc[i+1, 'netCashProvidedByOperatingActivities']
                r_cashflow.loc[i, 'acquisitionsNet'] = r_cashflow.loc[i, 'acquisitionsNet'] - r_cashflow.loc[i+1, 'acquisitionsNet']
                r_cashflow.loc[i, 'purchasesOfInvestments'] = r_cashflow.loc[i, 'purchasesOfInvestments'] - r_cashflow.loc[i+1, 'purchasesOfInvestments']
                r_cashflow.loc[i, 'salesMaturitiesOfInvestments'] = r_cashflow.loc[i, 'salesMaturitiesOfInvestments'] - r_cashflow.loc[i+1, 'salesMaturitiesOfInvestments']
                r_cashflow.loc[i, 'otherInvestingActivites'] = r_cashflow.loc[i, 'otherInvestingActivites'] - r_cashflow.loc[i+1, 'otherInvestingActivites']
                r_cashflow.loc[i, 'debtRepayment'] = r_cashflow.loc[i, 'debtRepayment'] - r_cashflow.loc[i+1, 'debtRepayment']
                r_cashflow.loc[i, 'commonStockIssued'] = r_cashflow.loc[i, 'commonStockIssued'] - r_cashflow.loc[i+1, 'commonStockIssued']
                r_cashflow.loc[i, 'commonStockRepurchased'] = r_cashflow.loc[i, 'commonStockRepurchased'] - r_cashflow.loc[i+1, 'commonStockRepurchased']
                r_cashflow.loc[i, 'dividendsPaid'] = r_cashflow.loc[i, 'dividendsPaid'] - r_cashflow.loc[i+1, 'dividendsPaid']
                r_cashflow.loc[i, 'otherFinancingActivites'] = r_cashflow.loc[i, 'otherFinancingActivites'] - r_cashflow.loc[i+1, 'otherFinancingActivites']
                r_cashflow.loc[i, 'netCashUsedProvidedByFinancingActivities'] = r_cashflow.loc[i, 'netCashUsedProvidedByFinancingActivities'] - r_cashflow.loc[i+1, 'netCashUsedProvidedByFinancingActivities']
                r_cashflow.loc[i, 'operatingCashFlow'] = r_cashflow.loc[i, 'operatingCashFlow'] - r_cashflow.loc[i+1, 'operatingCashFlow']
                r_cashflow.loc[i, 'capitalExpenditure'] = r_cashflow.loc[i, 'capitalExpenditure'] - r_cashflow.loc[i+1, 'capitalExpenditure']
                r_cashflow.loc[i, 'freeCashFlow'] = r_cashflow.loc[i, 'freeCashFlow'] - r_cashflow.loc[i+1, 'freeCashFlow']

        try:
            r_cashflow = r_cashflow.drop(r_cashflow.index[-1])
        except:
            pass
        start_date_f = end_date_f
        start_date_f = get_pre_quarter_end_date(start_date_f)
        start_date_f = datetime.strptime(start_date_f, "%Y%m%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")
        start_date_f = datetime.strptime(start_date_f, "%Y-%m-%d")
        start_date_f = start_date_f.strftime("%Y-%m-%d")

        ten_years_later = datetime.strptime(start_date_f, "%Y-%m-%d") + timedelta(days=3652)
        ten_years_later = ten_years_later.strftime("%Y-%m-%d")
        end_date_f = ten_years_later
        cashflow_data = pd.concat([cashflow_data, r_cashflow], ignore_index=True)
        # print(cashflow_data)
    cashflow_data = cashflow_data.drop_duplicates(subset='endDate')
    cashflow_data = cashflow_data.sort_values(by='endDate', ascending=False)
    cashflow_data = cashflow_data.reset_index(drop=True)
    cashflow_data = cashflow_data.fillna(0)
    # print(cashflow_data)
    return cashflow_data

def write_cashflow_data_into_table(stock_symbol_data):
    cashflow_exchange_name = 'cashflow_a'
    #建立连接
    metadata = MetaData()
    cashflow_table = Table(cashflow_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    # 获取表的列顺序
    inspector = inspect(engine)
    columns = inspector.get_columns(cashflow_exchange_name)
    column_names = [column['name'] for column in columns]
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(cashflow_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        ipo_date = str(stock_data['ipoDate'])
        fs_type = stock_data['fsType']
        if ipo_date == '':
            ipo_date = '19000101'
        ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
        last_date = conn.execute(f"SELECT MAX(endDate) FROM {cashflow_exchange_name} WHERE symbol = '{symbol}'")
        last_date = last_date.scalar()
        if last_date is None:
            last_date = ipo_date
        else:
            last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
        date_format = "%Y-%m-%d"
        last_date = datetime.strptime(last_date, date_format)
        # last_date = last_date + timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        if fs_type != 'non_financial':
            continue
        cashflow_data = get_symbol_cashflow_data(symbol, last_date, fs_type)
        try:
            cashflow_data = cashflow_data[column_names]
        except:
            continue
        insert_num = 0
        for index, row in cashflow_data.iterrows():
            data = row.to_dict()
            try:
                conn.execute(cashflow_table.insert(), data)
                insert_num += 1
            except:
                pass
        print('insert_num:' + str(insert_num))

def write_exchange_cashflow_data():
    exchange = 'a'
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    cashflow_exchange_name = 'cashflow_' + exchange
    if cashflow_exchange_name in table_names:
        # 创建元数据对象
        metadata = MetaData()
        # 如果表格存在，则输出提示信息
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入cashflow_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(cashflow_exchange_name + ' data updated!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(cashflow_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('deferredIncomeTax', String(50)),
                      Column('stockBasedCompensation', String(50)),
                      Column('changeInWorkingCapital', String(50)),
                      Column('accountsReceivables', String(50)),
                      Column('accountsPayables', String(50)),
                      Column('otherWorkingCapital', String(50)),
                      Column('otherNonCashItems', String(50)),
                      Column('netCashProvidedByOperatingActivities', String(50)),
                      Column('investmentsInPropertyPlantAndEquipment', String(50)),
                      Column('acquisitionsNet', String(50)),
                      Column('purchasesOfInvestments', String(50)),
                      Column('salesMaturitiesOfInvestments', String(50)),
                      Column('otherInvestingActivites', String(50)),
                      Column('netCashUsedForInvestingActivites', String(50)),
                      Column('debtRepayment', String(50)),
                      Column('commonStockIssued', String(50)),
                      Column('commonStockRepurchased', String(50)),
                      Column('dividendsPaid', String(50)),
                      Column('otherFinancingActivites', String(50)),
                      Column('netCashUsedProvidedByFinancingActivities', String(50)),
                      Column('effectOfForexChangesOnCash', String(50)),
                      Column('netChangeInCash', String(50)),
                      Column('cashAtEndOfPeriod', String(50)),
                      Column('cashAtBeginningOfPeriod', String(50)),
                      Column('operatingCashFlow', String(50)),
                      Column('capitalExpenditure', String(50)),
                      Column('freeCashFlow', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入cashflow_exchange_name
        split_num = int(len(stock_symbol_data) * 0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2 * split_num]
        stock_symbol_data_2 = stock_symbol_data[2 * split_num:]
        print(stock_symbol_data)
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_cashflow_data_into_table, args=(stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        print(cashflow_exchange_name + ' data inited!')

def init_exchange_indicator_data():
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    indicator_exchange_name = 'indicator_a'
    # 创建元数据对象
    metadata = MetaData()
    if indicator_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(indicator_exchange_name + ' Table exists')
    else:
        # 创建表对象
        table = Table(indicator_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('numberOfShares', String(50)),
                      Column('dividend', String(50)),
                      Column('netAssetValuePerShare', String(50)),
                      Column('revenuePerShare', String(50)),
                      Column('cashFlowPerShare', String(50)),
                      Column('debtToEquityRatio', String(50)),
                      Column('quickRatio', String(50)),
                      Column('earningsMultiple', String(50)),##就是ebitdaratio
                      Column('operatingMargin', String(50)),  ##就是operatingIncomeRatio
                      Column('pretaxProfitMargin', String(50)),  ##就是incomeBeforeTaxRatio
                      Column('netProfitMargin', String(50)),  ##就是netIncomeRatio
                      Column('grossProfitMargin', String(50)), ##就是grossProfitRatio
                      Column('roe', String(50)),
                      Column('dcfPerShare', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        print(indicator_exchange_name + ' Table created!')
        #从mysql中读取交易所财务三张表和分红数据
        Session = sessionmaker(bind=engine)
        session = Session()
        income_table_name = 'income_a'
        income_sql = "SELECT * FROM " + income_table_name
        income_data = pd.read_sql(income_sql, session.connection())
        balance_table_name = 'balance_a'
        balance_sql = "SELECT * FROM " + balance_table_name
        balance_data = pd.read_sql(balance_sql, session.connection())
        cashflow_table_name = 'cashflow_a'
        cashflow_sql = "SELECT * FROM " + cashflow_table_name
        cashflow_data = pd.read_sql(cashflow_sql, session.connection())
        dividend_table_name = 'dividend_a'
        dividend_sql = "SELECT * FROM " + dividend_table_name
        dividend_data = pd.read_sql(dividend_sql, session.connection())

        #合并三张表和分红表
        merge_data = pd.merge(dividend_data, income_data, on=['symbol', 'endDate'], how='outer')
        merge_data = pd.merge(merge_data, balance_data, on=['symbol', 'endDate'], how='outer')
        merge_data = pd.merge(merge_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
        merge_data = merge_data.dropna()
        merge_data = merge_data.reset_index(drop=True)

        indicator_data = pd.DataFrame()
        indicator_data['symbol'] = merge_data['symbol']
        indicator_data['endDate'] = merge_data['endDate']
        indicator_data['numberOfShares'] = merge_data['numberOfShares']
        indicator_data['dividend'] = merge_data['dividend']
        indicator_data['netAssetValuePerShare'] = merge_data['totalAssets'].astype(float) - \
                                                  merge_data['totalDebt'].astype(float)
        indicator_data['netAssetValuePerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         indicator_data['netAssetValuePerShare'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['revenuePerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         merge_data['revenue'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['cashFlowPerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         merge_data['operatingCashFlow'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['debtToEquityRatio'] = np.where(merge_data['totalStockholdersEquity'].astype(float) != 0,
                                         merge_data['totalDebt'].astype(float) /
                                         merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['quickRatio'] = (merge_data['totalCurrentAssets'].astype(float) -
                                        merge_data['inventory'].astype(float) -
                                        merge_data['deferredTaxLiabilitiesNonCurrent'].astype(float))
        indicator_data['quickRatio'] = np.where(merge_data['totalCurrentLiabilities'].astype(float) != 0,
                                                 indicator_data['quickRatio'].astype(float) /
                                                 merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['earningsMultiple'] = merge_data['ebitdaratio']
        indicator_data['operatingMargin'] = merge_data['operatingIncomeRatio']
        indicator_data['pretaxProfitMargin'] = merge_data['incomeBeforeTaxRatio']
        indicator_data['netProfitMargin'] = merge_data['netIncomeRatio']
        indicator_data['grossProfitMargin'] = merge_data['grossProfitRatio']
        indicator_data['roe'] = np.where(merge_data['totalStockholdersEquity'].astype(float) != 0,
                                         merge_data['netIncome'].astype(float) /
                                         merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['dcfPerShare'] = 0.0##这个地方是重点，我们现在先不给，后面长线系统会专门处理
        indicator_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
        indicator_data = indicator_data.reset_index(drop=True)
        indicator_data = indicator_data.replace([np.inf, -np.inf], np.nan)
        indicator_data = indicator_data.fillna(0)
        indicator_data.to_sql(indicator_exchange_name, con=engine, if_exists='append', index=False)
        print(indicator_exchange_name + ' data inited!')

def write_exchange_indicator_data():
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    indicator_exchange_name = 'indicator_a'
    # 创建元数据对象
    metadata = MetaData()
    if indicator_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        indicator_sql = "SELECT * FROM " + indicator_exchange_name
        indicator_exchange_data = pd.read_sql(indicator_sql, session.connection())
        #从mysql中读取交易所财务三张表和分红数据
        Session = sessionmaker(bind=engine)
        session = Session()
        income_table_name = 'income_a'
        income_sql = "SELECT * FROM " + income_table_name
        income_data = pd.read_sql(income_sql, session.connection())
        balance_table_name = 'balance_a'
        balance_sql = "SELECT * FROM " + balance_table_name
        balance_data = pd.read_sql(balance_sql, session.connection())
        cashflow_table_name = 'cashflow_a'
        cashflow_sql = "SELECT * FROM " + cashflow_table_name
        cashflow_data = pd.read_sql(cashflow_sql, session.connection())
        dividend_table_name = 'dividend_a'
        dividend_sql = "SELECT * FROM " + dividend_table_name
        dividend_data = pd.read_sql(dividend_sql, session.connection())

        #合并三张表和分红表
        merge_data = pd.merge(dividend_data, income_data, on=['symbol', 'endDate'], how='outer')
        merge_data = pd.merge(merge_data, balance_data, on=['symbol', 'endDate'], how='outer')
        merge_data = pd.merge(merge_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
        merge_data = merge_data.dropna()
        merge_data = merge_data.reset_index(drop=True)

        #只保留新的值
        mask = (merge_data['symbol'].isin(indicator_exchange_data['symbol'])) & (merge_data['endDate'].isin(indicator_exchange_data['endDate']))
        merge_data = merge_data[~mask]
        merge_data = merge_data.reset_index(drop=True)

        #生成indicator数据
        indicator_data = pd.DataFrame()
        indicator_data['symbol'] = merge_data['symbol']
        indicator_data['endDate'] = merge_data['endDate']
        indicator_data['numberOfShares'] = merge_data['numberOfShares']
        indicator_data['dividend'] = merge_data['dividend']
        indicator_data['netAssetValuePerShare'] = merge_data['totalAssets'].astype(float) - \
                                                  merge_data['totalDebt'].astype(float)
        indicator_data['netAssetValuePerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         indicator_data['netAssetValuePerShare'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['revenuePerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         merge_data['revenue'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['cashFlowPerShare'] = np.where(merge_data['numberOfShares'].astype(float) != 0,
                                         merge_data['operatingCashFlow'].astype(float) /
                                         merge_data['numberOfShares'].astype(float), 0)
        indicator_data['debtToEquityRatio'] = np.where(merge_data['totalStockholdersEquity'].astype(float) != 0,
                                         merge_data['totalDebt'].astype(float) /
                                         merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['quickRatio'] = (merge_data['totalCurrentAssets'].astype(float) -
                                        merge_data['inventory'].astype(float) -
                                        merge_data['deferredTaxLiabilitiesNonCurrent'].astype(float))
        indicator_data['quickRatio'] = np.where(merge_data['totalCurrentLiabilities'].astype(float) != 0,
                                                 indicator_data['quickRatio'].astype(float) /
                                                 merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['earningsMultiple'] = merge_data['ebitdaratio']
        indicator_data['operatingMargin'] = merge_data['operatingIncomeRatio']
        indicator_data['pretaxProfitMargin'] = merge_data['incomeBeforeTaxRatio']
        indicator_data['netProfitMargin'] = merge_data['netIncomeRatio']
        indicator_data['grossProfitMargin'] = merge_data['grossProfitRatio']
        indicator_data['roe'] = np.where(merge_data['totalStockholdersEquity'].astype(float) != 0,
                                         merge_data['netIncome'].astype(float) /
                                         merge_data['totalStockholdersEquity'].astype(float), 0)
        indicator_data['dcfPerShare'] = 0.0##这个地方是重点，我们现在先不给，后面长线系统会专门处理
        indicator_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
        indicator_data = indicator_data.reset_index(drop=True)
        indicator_data = indicator_data.replace([np.inf, -np.inf], np.nan)
        indicator_data = indicator_data.fillna(0)
        # print(indicator_data)
        indicator_data.to_sql(indicator_exchange_name, con=engine, if_exists='append', index=False)

        # #建立连接
        # metadata = MetaData()
        # indicator_table = Table(indicator_exchange_name, metadata, autoload=True, autoload_with=engine)
        # conn = engine.connect()
        # for index, row in indicator_data.iterrows():
        #     data = row.to_dict()
        #     try:
        #         conn.execute(indicator_table.insert(), data)
        #         print(data['symbol'] + ' ' + data['endDate'] + ' indicator_data inserted!')
        #     except:
        #         pass
    else:
        print(indicator_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_indicator_data()
def write_a_data():
    write_exchange_stock_symbol_data()
    write_exchange_income_data()
    write_exchange_balance_data()
    write_exchange_cashflow_data()
    write_exchange_dividend_data()
    write_exchange_indicator_data()
    write_exchange_daily_data()

if __name__ == "__main__":
    # write_exchange_stock_symbol_data()
    # write_exchange_income_data()
    # write_exchange_balance_data()
    # write_exchange_cashflow_data()
    # write_exchange_dividend_data()
    # write_exchange_indicator_data()
    # write_exchange_daily_data()
    # write_exchange_dividend_data()
    # write_exchange_daily_data()
    # get_symbol_income_data('000001.sz', '2000-09-30', 'bank')
    # data.to_csv('company.csv', index=False, encoding="utf_8_sig")
    # date_str = '20231030'
    # # last_end = get_pre_quarter_end_dat