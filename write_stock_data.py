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

# from bs4 import BeautifulSoup

import calendar
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
token ='6263d89930d3ba4b1329d603814270ad'

##
exchange_all = ['SIX', 'SHZ', 'BUD', 'BRU', 'KUW', 'TSX', 'NZE', 'FKA', 'KOE', 'JNB', 'NASDAQ', 'ETF', 'SGO', 'MCE',
                'BSE', 'DUS', 'JKT', 'STO', 'HEL', 'ASE', 'IST', 'PNK', 'TAL', 'OSL', 'SES', 'JPX', 'TLV', 'DOH', 'ISE',
                'SAT', 'MEX', 'LIS', 'SAU', 'EURONEXT', 'VIE', 'MUN', 'AMEX', 'OTC', 'SHH', 'HAM', 'ATH', 'KLS', 'SAO',
                'SET', 'CNQ', 'ASX', 'KSC', 'IOB', 'NEO', 'CPH', 'WSE', 'HKSE', 'LSE', 'NYSE', 'RIS', 'ICE',
                'NSE', 'TAI', 'MCX', 'DFM', 'TWO', 'XETRA', 'PRA', 'AMS', 'BUE', 'KOSDAQ', 'STU', 'MIL']

#exhange_reference_dict
china_reference_dict = {'Shenzhen':['SZ', 'SHZ', 'SZSE', 'Shenzhen'],
                        'Shanghai':['SS', 'SHH', 'SSE', 'Shanghai'] }
exchange_reference_dict = {'Shenzhen':['SZ', 'SHZ', 'SZSE', 'Shenzhen'],
                           'Shanghai':['SS', 'SHH', 'SSE', 'Shanghai'],
                           'Swiss':['SW', 'SIX', 'SWX', 'Swiss Exchange'],
                           'Toronto':['TO', 'TSX', 'TSX', 'Toronto'],
                           'Johannesburg':['JO', 'JNB', 'JSE', 'Johannesburg'],
                           'Jakarta':['JK', 'JKT', 'JKSE', 'Jakarta Stock Exchange'],
                           'Stockholm':['ST', 'STO', 'STO', 'Stockholm Stock Exchange'],
                           'Oslo':['OL', 'OSL', 'OSE', 'Oslo Stock Exchange'],
                           'Tokyo':['T', 'JPX', 'JPX', 'Tokyo'],
                           'Saudi':['SR', 'SAU', 'Tadawul', 'Saudi'],
                           'Brussels':['BR', 'EURONEXT', 'Euronext', 'Brussels'],
                           'Arca':['', 'AMEX', 'Arca', 'New York Stock Exchange Arca'],
                           'SaoPaulo':['SA', 'SAO', 'Bovespa', 'São Paulo'],
                           'Thailand':['BK', 'SET', 'SET', 'Thailand'],
                           'Canadian':['CN', 'CNQ', 'CSE', 'Canadian Sec'],
                           'Australian':['AX', 'ASX', 'ASX', 'Australian Securities Exchange'],
                           'Korea':['KS', 'KSC', 'KSE', 'KSE'],
                           'Warsaw':['WA', 'WSE', 'WSE', 'Warsaw Stock Exchange'],
                           'HongKong':['HK', 'HKSE', 'HKEx', 'HKSE'],
                           'London':['L', 'LSE', 'LSE', 'London Stock Exchange'],
                           'NewYork':['', 'NYSE', 'NYSE', 'New York Stock Exchange'],
                           'India':['NS', 'NSE', 'NSE', 'National Stock Exchange of India'],
                           'Taiwan':['TW', 'Tai', 'TWSE', 'Taiwan'],
                           'Moscow':['ME', 'MCX', 'MOEX', 'MCX'],
                           'Frankfurt':['DE', 'XETRA', 'FWB', 'Frankfurt Stock Exchange'],
                           'Milan':['MI', 'MIL', 'MIL', 'Milan'],
                           'Nasdaq':['', 'NASDAQ', 'NASDAQ', 'Nasdaq'],
                           'Pnk':['', 'PNK', 'PNK', 'Other OTC']}


candidate_exchanges = ['SIX', 'SHZ', 'TSX', 'JNB', 'NASDAQ', 'JKT', 'STO', 'PNK', 'OSL', 'JPX', 'TLV', 'SAU',
                       'EURONEXT', 'AMEX', 'OTC', 'SHH', 'SAO', 'SET', 'CNQ', 'ASX', 'KSC', 'WSE', 'HKSE', 'LSE',
                       'NYSE', 'NSE', 'TAI', 'MCX', 'XETRA', 'MIL']
#创建数据库
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock', pool_size=10, max_overflow=20)

def get_jsonparsed_data(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def write_symbol_data_into_table(exchange, stock_list):
    #建立连接
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    metadata = MetaData()
    stock_table = Table(exchange_stock_symbol_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    company_list = []
    for i in range(len(stock_list)):
        print(exchange_stock_symbol_name + ':' + str(float(i)/(len(stock_list)+1.0)))
        symbol = stock_list[i]
        try:
            url = 'https://financialmodelingprep.com/api/v3/profile/' + symbol + '?apikey=' + token
            company_profile = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        if len(company_profile) > 0:
            company_profile = company_profile[0]
            company_profile.pop('phone')
            company_profile.pop('fullTimeEmployees')
            company_profile.pop('country')
            company_profile.pop('exchangeShortName')
            company_profile.pop('price')
            company_profile.pop('beta')
            company_profile.pop('volAvg')
            company_profile.pop('mktCap')
            company_profile.pop('lastDiv')
            company_profile.pop('range')
            company_profile.pop('changes')
            company_profile.pop('cik')
            company_profile.pop('isin')
            company_profile.pop('cusip')
            company_profile.pop('industry')
            company_profile.pop('description')
            company_profile.pop('ceo')
            company_profile.pop('sector')
            company_profile.pop('address')
            company_profile.pop('state')
            company_profile.pop('zip')
            company_profile.pop('dcfDiff')
            company_profile.pop('dcf')
            company_profile.pop('image')
            company_profile.pop('defaultImage')
            company_profile.pop('isEtf')
            company_profile.pop('isActivelyTrading')
            company_profile.pop('isAdr')
            company_profile.pop('isFund')
            # print(company_profile)
            company_list.append(company_profile)
    stock_data = pd.DataFrame.from_records(company_list)
    stock_data['exchange'] = exchange_reference_dict[exchange][3]
    stock_data = stock_data.rename(columns={'companyName': 'name'})
    #修改symbol后缀为实际的交易所简写
    symbol_split = stock_data['symbol'][0].split('.')
    if len(symbol_split) > 1:
        symbol_suffix_pre = symbol_split[1]
        stock_data['financialmodelingprep_symbol'] = stock_data['symbol']
        stock_data['symbol'] = stock_data['symbol'].str.replace(symbol_suffix_pre, exchange_reference_dict[exchange][2])
    else:
        stock_data['financialmodelingprep_symbol'] = stock_data['symbol']
        stock_data['symbol'] = stock_data['symbol'] + '.' + exchange_reference_dict[exchange][2]
    stock_data['tushare_symbol'] = stock_data['symbol']
    stock_data['name'] = stock_data['name'].fillna('')
    stock_data['city'] = stock_data['city'].fillna('')
    stock_data['website'] = stock_data['website'].fillna('https://')
    stock_data['ipoDate'] = stock_data['ipoDate'].fillna('1900-01-01')
    stock_data['ipoDate'] = stock_data['ipoDate'].apply(lambda x: x.replace('-', ''))
    stock_data = stock_data.drop_duplicates(subset=['symbol'], keep='first')
    stock_data.to_sql(exchange_stock_symbol_name, con=engine, if_exists='append', index=False)
    print(stock_data)

def init_exchange_stock_symbol_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    if exchange_stock_symbol_name in table_names:
        # 如果表格存在，则输出提示信息
        print(exchange_stock_symbol_name + ' table exists!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(exchange_stock_symbol_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('name', String(200)),
                      Column('currency', String(50)),
                      Column('exchange', String(50)),
                      Column('website', String(200)),
                      Column('city', String(50)),
                      Column('ipoDate', String(50)),
                      Column('financialmodelingprep_symbol', String(50)),
                      Column('tushare_symbol', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        exchange_reference = exchange_reference_dict[exchange][1]
        url = 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=' + token
        symbol_list = get_jsonparsed_data(url)

        url = 'https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&exchange=' + exchange_reference + '&apikey=' + token
        stock = get_jsonparsed_data(url)
        stock = pd.DataFrame.from_records(stock)
        stock = stock['symbol'].tolist()
        print(len(stock))
        stock_list = list(set(symbol_list) & set(stock))
        # stock_list = [item for item in stock_list if item.endswith(exchange_reference_dict[exchange][0])]
        print(len(stock_list))
        symbol_num = len(stock_list)
        split_num = int(0.333*symbol_num)
        stock_list0 = stock_list[:split_num]
        stock_list1 = stock_list[split_num:2*split_num]
        stock_list2 = stock_list[2*split_num:]

        t0 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list0))
        t1 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list1))
        t2 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list2))
        t0.start()
        t1.start()
        t2.start()
        t0.join()
        t1.join()
        t2.join()
        print(exchange_stock_symbol_name + ' data inited!')

def write_exchange_stock_symbol_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    if exchange_stock_symbol_name in table_names:
        metadata = MetaData()
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data_pre = pd.read_sql(stock_symbol_sql, session.connection())
        symbol_list_pre = stock_symbol_data_pre['financialmodelingprep_symbol'].tolist()
        #建立链接
        stock_symbol_table = Table(exchange_stock_symbol_name, metadata, autoload=True, autoload_with=engine)
        conn = engine.connect()
        #获得最新的exchange内的所有symbol数据
        exchange_reference = exchange_reference_dict[exchange][1]
        url = 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=' + token
        symbol_list = get_jsonparsed_data(url)
        url = 'https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&exchange=' + exchange_reference + '&apikey=' + token
        stock = get_jsonparsed_data(url)
        stock = pd.DataFrame.from_records(stock)
        stock = stock['symbol'].tolist()
        stock_list = list(set(symbol_list) & set(stock))
        print(len(stock_list))
        stock_list = list(set(stock_list) - set(symbol_list_pre))
        # print(len(stock_list))
        # stock_list = [item for item in stock_list if item.endswith(exchange_reference_dict[exchange][0])]
        print(len(stock_list))
        print(stock_list)
        company_list = []
        for i in range(len(stock_list)):
            print(exchange_stock_symbol_name + ':' + str(float(i) / (len(stock_list) + 1.0)))
            symbol = stock_list[i]
            try:
                url = 'https://financialmodelingprep.com/api/v3/profile/' + symbol + '?apikey=' + token
                company_profile = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(company_profile) > 0:
                company_profile = company_profile[0]
                company_profile.pop('phone')
                company_profile.pop('fullTimeEmployees')
                company_profile.pop('country')
                company_profile.pop('exchangeShortName')
                company_profile.pop('price')
                company_profile.pop('beta')
                company_profile.pop('volAvg')
                company_profile.pop('mktCap')
                company_profile.pop('lastDiv')
                company_profile.pop('range')
                company_profile.pop('changes')
                company_profile.pop('cik')
                company_profile.pop('isin')
                company_profile.pop('cusip')
                company_profile.pop('industry')
                company_profile.pop('description')
                company_profile.pop('ceo')
                company_profile.pop('sector')
                company_profile.pop('address')
                company_profile.pop('state')
                company_profile.pop('zip')
                company_profile.pop('dcfDiff')
                company_profile.pop('dcf')
                company_profile.pop('image')
                company_profile.pop('defaultImage')
                company_profile.pop('isEtf')
                company_profile.pop('isActivelyTrading')
                company_profile.pop('isAdr')
                company_profile.pop('isFund')
                company_list.append(company_profile)
        if len(company_list) > 0:
            stock_data = pd.DataFrame.from_records(company_list)
            stock_data['exchange'] = exchange_reference_dict[exchange][3]
            stock_data = stock_data.rename(columns={'companyName': 'name'})
            # 修改symbol后缀为实际的交易所简写
            symbol_split = stock_data['symbol'][0].split('.')
            if len(symbol_split) > 1:
                symbol_suffix_pre = symbol_split[1]
                stock_data['financialmodelingprep_symbol'] = stock_data['symbol']
                stock_data['symbol'] = stock_data['symbol'].str.replace(symbol_suffix_pre,
                                                                        exchange_reference_dict[exchange][2])
            else:
                stock_data['financialmodelingprep_symbol'] = stock_data['symbol']
                stock_data['symbol'] = stock_data['symbol'] + '.' + exchange_reference_dict[exchange][2]

            stock_data['tushare_symbol'] = stock_data['symbol']
            stock_data['name'] = stock_data['name'].fillna('')
            stock_data['city'] = stock_data['city'].fillna('')
            stock_data['website'] = stock_data['website'].fillna('https://')
            stock_data['ipoDate'] = stock_data['ipoDate'].fillna('1900-01-01')
            stock_data['ipoDate'] = stock_data['ipoDate'].apply(lambda x: x.replace('-', ''))
            print(stock_data)
            for index, row in stock_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(stock_symbol_table.insert(), data)
                    print(data['symbol'] + ' symbol_data inserted!')
                except:
                    pass
    else:
        print(exchange_stock_symbol_name + ' table not exist!')
        print('start initing table...')
        init_exchange_stock_symbol_data(exchange)

def init_exchange_income_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    income_exchange_name = 'income_' + exchange
    if income_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(income_exchange_name + ' Table exists')
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
        #再插入income数据
        for i in range(len(stock_symbol_data)):
            print(income_exchange_name + ':'  + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            try:
                url = 'https://financialmodelingprep.com/api/v3/income-statement/'+fmp_symbol+'?period=quarter&limit=480&apikey='+token
                income_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(income_data) != 0:
                income_data = pd.DataFrame(income_data)
                income_data = income_data.drop('symbol', axis=1)
                income_data.insert(0, 'symbol', [symbol]*len(income_data))
                income_data = income_data.rename(columns={'fillingDate': 'ann_date'})
                income_data = income_data.drop('cik', axis=1)
                income_data = income_data.drop('acceptedDate', axis=1)
                income_data = income_data.drop('calendarYear', axis=1)
                income_data = income_data.drop('period', axis=1)
                income_data = income_data.drop('link', axis=1)
                income_data = income_data.drop('finalLink', axis=1)
                col1_index = income_data.columns.get_loc('reportedCurrency')
                col2_index = income_data.columns.get_loc('ann_date')
                income_data.iloc[:, [col2_index, col1_index]] = income_data.iloc[:, [col1_index, col2_index]]
                income_data = income_data.rename(columns={'date': 'endDate'})
                income_data = income_data.rename(columns={'reportedCurrency': 'temp'})
                income_data = income_data.rename(columns={'ann_date': 'reportedCurrency'})
                income_data = income_data.rename(columns={'temp': 'ann_date'})
                income_data['endDate'] = income_data['endDate'].apply(lambda x: x.replace('-', ''))
                income_data = income_data.drop('ann_date', axis=1)
                income_data = income_data.drop('reportedCurrency', axis=1)
                income_data = income_data.sort_values(by='endDate', ascending=False)
                income_data = income_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
                income_data = income_data.reset_index(drop=True)
                income_data = income_data.fillna(0)
                income_data.to_sql(income_exchange_name, con=engine, if_exists='append', index=False)
        print(income_exchange_name + ' data inited!')

def write_exchange_income_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    income_exchange_name = 'income_' + exchange
    if income_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        income_table = Table(income_exchange_name, metadata, autoload=True, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(income_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            ipo_date = str(stock_data['ipoDate'])
            if ipo_date == '':
                ipo_date = '19000101'
            ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
            last_date = conn.execute(f"SELECT MAX(endDate) FROM {income_exchange_name} WHERE symbol = '{symbol}'")
            last_date = last_date.scalar()
            if last_date is None:
                last_date = ipo_date
            else:
                last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
            endDate = last_date.replace('-', '')
            date_format = "%Y-%m-%d"
            last_date = datetime.datetime.strptime(last_date, date_format)
            last_date = last_date + datetime.timedelta(days=1)
            today = datetime.datetime.today()
            # 计算日期差
            delta = today - last_date
            quarter_num = int(float(delta.days/90))
            if quarter_num < 1:
                continue
            try:
                url = 'https://financialmodelingprep.com/api/v3/income-statement/' + fmp_symbol + '?period=quarter&limit=' + str(quarter_num) + '&apikey=' + token
                income_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(income_data) == 0:
                continue
            income_data = pd.DataFrame(income_data)
            income_data = income_data[income_data['date'] > endDate]
            if len(income_data) == 0:
                continue
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.drop('symbol', axis=1)
            income_data.insert(0, 'symbol', [symbol]*len(income_data))
            income_data = income_data.rename(columns={'fillingDate': 'ann_date'})
            income_data = income_data.drop('cik', axis=1)
            income_data = income_data.drop('acceptedDate', axis=1)
            income_data = income_data.drop('calendarYear', axis=1)
            income_data = income_data.drop('period', axis=1)
            income_data = income_data.drop('link', axis=1)
            income_data = income_data.drop('finalLink', axis=1)
            col1_index = income_data.columns.get_loc('reportedCurrency')
            col2_index = income_data.columns.get_loc('ann_date')
            income_data.iloc[:, [col2_index, col1_index]] = income_data.iloc[:, [col1_index, col2_index]]
            income_data = income_data.rename(columns={'date': 'endDate'})
            income_data = income_data.rename(columns={'reportedCurrency': 'temp'})
            income_data = income_data.rename(columns={'ann_date': 'reportedCurrency'})
            income_data = income_data.rename(columns={'temp': 'ann_date'})
            income_data['endDate'] = income_data['endDate'].apply(lambda x: x.replace('-', ''))
            income_data = income_data.drop('ann_date', axis=1)
            income_data = income_data.drop('reportedCurrency', axis=1)
            income_data = income_data.sort_values(by='endDate', ascending=False)
            income_data = income_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0)
            for index, row in income_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(income_table.insert(), data)
                    print(symbol + ' ' + data['endDate'] + ' income_data inserted!')
                except:
                    print('error')
                    pass
    else:
        print(income_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_income_data(exchange)

def init_exchange_balance_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    balance_exchange_name = 'balance_' + exchange
    if balance_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(balance_exchange_name + ' Table exists')
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
        #再插入balance数据
        for i in range(len(stock_symbol_data)):
            print(balance_exchange_name + ':'  + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            try:
                url = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+fmp_symbol+'?period=quarter&limit=480&apikey='+token
                balance_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(balance_data) != 0:
                balance_data = pd.DataFrame(balance_data)
                balance_data = balance_data.drop('symbol', axis=1)
                balance_data.insert(0, 'symbol', [symbol]*len(balance_data))
                balance_data = balance_data.rename(columns={'fillingDate': 'ann_date'})
                balance_data = balance_data.drop('cik', axis=1)
                balance_data = balance_data.drop('acceptedDate', axis=1)
                balance_data = balance_data.drop('calendarYear', axis=1)
                balance_data = balance_data.drop('period', axis=1)
                balance_data = balance_data.drop('link', axis=1)
                balance_data = balance_data.drop('finalLink', axis=1)
                col1_index = balance_data.columns.get_loc('reportedCurrency')
                col2_index = balance_data.columns.get_loc('ann_date')
                balance_data.iloc[:, [col2_index, col1_index]] = balance_data.iloc[:, [col1_index, col2_index]]
                balance_data = balance_data.rename(columns={'date': 'endDate'})
                balance_data = balance_data.rename(columns={'reportedCurrency': 'temp'})
                balance_data = balance_data.rename(columns={'ann_date': 'reportedCurrency'})
                balance_data = balance_data.rename(columns={'temp': 'ann_date'})
                balance_data['endDate'] = balance_data['endDate'].apply(lambda x: x.replace('-', ''))
                balance_data = balance_data.drop('ann_date', axis=1)
                balance_data = balance_data.drop('reportedCurrency', axis=1)
                balance_data = balance_data.sort_values(by='endDate', ascending=False)
                balance_data = balance_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
                balance_data = balance_data.reset_index(drop=True)
                balance_data = balance_data.fillna(0)
                balance_data.to_sql(balance_exchange_name, con=engine, if_exists='append', index=False)
        print(balance_exchange_name + ' data inited!')

def write_exchange_balance_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    balance_exchange_name = 'balance_' + exchange
    if balance_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        balance_table = Table(balance_exchange_name, metadata, autoload=True, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(balance_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            ipo_date = str(stock_data['ipoDate'])
            if ipo_date == '':
                ipo_date = '19000101'
            ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
            last_date = conn.execute(f"SELECT MAX(endDate) FROM {balance_exchange_name} WHERE symbol = '{symbol}'")
            last_date = last_date.scalar()
            if last_date is None:
                last_date = ipo_date
            else:
                last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
            endDate = last_date.replace('-', '')
            date_format = "%Y-%m-%d"
            last_date = datetime.datetime.strptime(last_date, date_format)
            last_date = last_date + datetime.timedelta(days=1)
            today = datetime.datetime.today()
            # 计算日期差
            delta = today - last_date
            quarter_num = int(float(delta.days/90))
            if quarter_num < 1:
                continue
            try:
                url = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+fmp_symbol+'?period=quarter&limit=' + str(quarter_num) + '&apikey=' + token
                balance_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(balance_data) == 0:
                continue
            balance_data = pd.DataFrame(balance_data)
            balance_data = balance_data[balance_data['date'] > endDate]
            if len(balance_data) == 0:
                continue
            balance_data = balance_data.reset_index(drop=True)
            balance_data = balance_data.drop('symbol', axis=1)
            balance_data.insert(0, 'symbol', [symbol]*len(balance_data))
            balance_data = balance_data.rename(columns={'fillingDate': 'ann_date'})
            balance_data = balance_data.drop('cik', axis=1)
            balance_data = balance_data.drop('acceptedDate', axis=1)
            balance_data = balance_data.drop('calendarYear', axis=1)
            balance_data = balance_data.drop('period', axis=1)
            balance_data = balance_data.drop('link', axis=1)
            balance_data = balance_data.drop('finalLink', axis=1)
            col1_index = balance_data.columns.get_loc('reportedCurrency')
            col2_index = balance_data.columns.get_loc('ann_date')
            balance_data.iloc[:, [col2_index, col1_index]] = balance_data.iloc[:, [col1_index, col2_index]]
            balance_data = balance_data.rename(columns={'date': 'endDate'})
            balance_data = balance_data.rename(columns={'reportedCurrency': 'temp'})
            balance_data = balance_data.rename(columns={'ann_date': 'reportedCurrency'})
            balance_data = balance_data.rename(columns={'temp': 'ann_date'})
            balance_data['endDate'] = balance_data['endDate'].apply(lambda x: x.replace('-', ''))
            balance_data = balance_data.drop('ann_date', axis=1)
            balance_data = balance_data.drop('reportedCurrency', axis=1)
            balance_data = balance_data.sort_values(by='endDate', ascending=False)
            balance_data = balance_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            balance_data = balance_data.reset_index(drop=True)
            balance_data = balance_data.fillna(0)
            for index, row in balance_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(balance_table.insert(), data)
                    print(symbol + ' ' + data['endDate'] + ' balance_data inserted!')
                except:
                    pass
    else:
        print(balance_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_balance_data(exchange)

def init_exchange_cashflow_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    cashflow_exchange_name = 'cashflow_' + exchange
    if cashflow_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(cashflow_exchange_name + ' Table exists')
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
        #再插入cashflow数据
        for i in range(len(stock_symbol_data)):
            print(cashflow_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            try:
                url = 'https://financialmodelingprep.com/api/v3/cash-flow-statement/'+fmp_symbol+'?period=quarter&limit=480&apikey='+token
                cashflow_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(cashflow_data) != 0:
                cashflow_data = pd.DataFrame(cashflow_data)
                cashflow_data = cashflow_data.drop('symbol', axis=1)
                cashflow_data.insert(0, 'symbol', [symbol]*len(cashflow_data))
                cashflow_data = cashflow_data.rename(columns={'fillingDate': 'ann_date'})
                cashflow_data = cashflow_data.drop('cik', axis=1)
                cashflow_data = cashflow_data.drop('acceptedDate', axis=1)
                cashflow_data = cashflow_data.drop('calendarYear', axis=1)
                cashflow_data = cashflow_data.drop('period', axis=1)
                cashflow_data = cashflow_data.drop('link', axis=1)
                cashflow_data = cashflow_data.drop('finalLink', axis=1)
                # 删除cashflow_data的netIncome，depreciationAndAmortization, inventory这3个数据在income_data或则balance_data存在了
                cashflow_data = cashflow_data.drop('netIncome', axis=1)
                cashflow_data = cashflow_data.drop('depreciationAndAmortization', axis=1)
                cashflow_data = cashflow_data.drop('inventory', axis=1)
                col1_index = cashflow_data.columns.get_loc('reportedCurrency')
                col2_index = cashflow_data.columns.get_loc('ann_date')
                cashflow_data.iloc[:, [col2_index, col1_index]] = cashflow_data.iloc[:, [col1_index, col2_index]]
                cashflow_data = cashflow_data.rename(columns={'date': 'endDate'})
                cashflow_data = cashflow_data.rename(columns={'reportedCurrency': 'temp'})
                cashflow_data = cashflow_data.rename(columns={'ann_date': 'reportedCurrency'})
                cashflow_data = cashflow_data.rename(columns={'temp': 'ann_date'})
                cashflow_data['endDate'] = cashflow_data['endDate'].apply(lambda x: x.replace('-', ''))
                cashflow_data = cashflow_data.drop('ann_date', axis=1)
                cashflow_data = cashflow_data.drop('reportedCurrency', axis=1)
                cashflow_data = cashflow_data.sort_values(by='endDate', ascending=False)
                cashflow_data = cashflow_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
                cashflow_data = cashflow_data.reset_index(drop=True)
                cashflow_data = cashflow_data.fillna(0)
                cashflow_data.to_sql(cashflow_exchange_name, con=engine, if_exists='append', index=False)
        print(cashflow_exchange_name + ' data inited!')

def write_exchange_cashflow_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    cashflow_exchange_name = 'cashflow_' + exchange
    if cashflow_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        cashflow_table = Table(cashflow_exchange_name, metadata, autoload=True, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(cashflow_exchange_name  + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
            ipo_date = str(stock_data['ipoDate'])
            if ipo_date == '':
                ipo_date = '19000101'
            ipo_date = "-".join([ipo_date[:4], ipo_date[4:6], ipo_date[6:]])
            last_date = conn.execute(f"SELECT MAX(endDate) FROM {cashflow_exchange_name} WHERE symbol = '{symbol}'")
            last_date = last_date.scalar()
            if last_date is None:
                last_date = ipo_date
            else:
                last_date = "-".join([last_date[:4], last_date[4:6], last_date[6:]])
            endDate = last_date.replace('-', '')
            date_format = "%Y-%m-%d"
            last_date = datetime.datetime.strptime(last_date, date_format)
            last_date = last_date + datetime.timedelta(days=1)
            today = datetime.datetime.today()
            # 计算日期差
            delta = today - last_date
            quarter_num = int(float(delta.days/90))
            if quarter_num < 1:
                continue
            try:
                url = 'https://financialmodelingprep.com/api/v3/cash-flow-statement/'+fmp_symbol+'?period=quarter&limit=' + str(quarter_num) + '&apikey=' + token
                cashflow_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(cashflow_data) == 0:
                continue
            cashflow_data = pd.DataFrame(cashflow_data)
            cashflow_data = cashflow_data[cashflow_data['date'] > endDate]
            if len(cashflow_data) == 0:
                continue
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.drop('symbol', axis=1)
            cashflow_data.insert(0, 'symbol', [symbol]*len(cashflow_data))
            cashflow_data = cashflow_data.rename(columns={'fillingDate': 'ann_date'})
            cashflow_data = cashflow_data.drop('cik', axis=1)
            cashflow_data = cashflow_data.drop('acceptedDate', axis=1)
            cashflow_data = cashflow_data.drop('calendarYear', axis=1)
            cashflow_data = cashflow_data.drop('period', axis=1)
            cashflow_data = cashflow_data.drop('link', axis=1)
            cashflow_data = cashflow_data.drop('finalLink', axis=1)
            # 删除cashflow_data的netIncome，depreciationAndAmortization, inventory这3个数据在income_data或则balance_data存在了
            cashflow_data = cashflow_data.drop('netIncome', axis=1)
            cashflow_data = cashflow_data.drop('depreciationAndAmortization', axis=1)
            cashflow_data = cashflow_data.drop('inventory', axis=1)
            col1_index = cashflow_data.columns.get_loc('reportedCurrency')
            col2_index = cashflow_data.columns.get_loc('ann_date')
            cashflow_data.iloc[:, [col2_index, col1_index]] = cashflow_data.iloc[:, [col1_index, col2_index]]
            cashflow_data = cashflow_data.rename(columns={'date': 'endDate'})
            cashflow_data = cashflow_data.rename(columns={'reportedCurrency': 'temp'})
            cashflow_data = cashflow_data.rename(columns={'ann_date': 'reportedCurrency'})
            cashflow_data = cashflow_data.rename(columns={'temp': 'ann_date'})
            cashflow_data['endDate'] = cashflow_data['endDate'].apply(lambda x: x.replace('-', ''))
            cashflow_data = cashflow_data.drop('ann_date', axis=1)
            cashflow_data = cashflow_data.drop('reportedCurrency', axis=1)
            cashflow_data = cashflow_data.sort_values(by='endDate', ascending=False)
            cashflow_data = cashflow_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.fillna(0)
            for index, row in cashflow_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(cashflow_table.insert(), data)
                    print(symbol + ' ' + data['endDate'] + ' cashflow_data inserted!')
                except:
                    pass
    else:
        print(cashflow_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_cashflow_data(exchange)

#求取上个季度的最后一天
def get_pre_quarter_end_date(date_str):
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str
def init_exchange_dividend_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    dividend_exchange_name = 'dividend_' + exchange.lower()
    # 创建元数据对象
    metadata = MetaData()
    if dividend_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(dividend_exchange_name + ' Table exists')
    else:
        # 创建表对象
        table = Table(dividend_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('numberOfShares', String(50)),
                      Column('dividend', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
    # 插入dividend数据
    # 获得所有exchange内的股票信息
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    Session = sessionmaker(bind=engine)
    session = Session()
    stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
    stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
    for i in range(len(stock_symbol_data)):
        print(dividend_exchange_name + ':' + str(float(i)/len(stock_symbol_data)))
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        fmp_symbol = stock_data['financialmodelingprep_symbol']
        # 获得基本的企业数据
        try:
            url = 'https://financialmodelingprep.com/api/v3/enterprise-values/' + fmp_symbol + '?period=quarter&limit=480&apikey=' + token
            enterprise_values_data = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        enterprise_values_data = pd.DataFrame(enterprise_values_data)
        if len(enterprise_values_data) == 0:
            continue
        enterprise_values_data['symbol'] = symbol
        enterprise_values_data['date'] = enterprise_values_data['date'].apply(lambda x: x.replace('-', ''))
        enterprise_values_data = enterprise_values_data.rename(columns={'date': 'endDate'})
        enterprise_values_data = enterprise_values_data.drop('stockPrice', axis=1)
        enterprise_values_data = enterprise_values_data.drop('marketCapitalization', axis=1)
        enterprise_values_data = enterprise_values_data.drop('minusCashAndCashEquivalents', axis=1)
        enterprise_values_data = enterprise_values_data.drop('addTotalDebt', axis=1)
        enterprise_values_data = enterprise_values_data.drop('enterpriseValue', axis=1)
        enterprise_values_data.drop_duplicates(subset=['endDate'], keep='first', inplace=True)
        enterprise_values_data = enterprise_values_data.reset_index(drop=True)
        # 获得分红数据
        try:
            url = 'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/' + fmp_symbol + '?period=quarter&limit=480&apikey=' + token
            dividend_data = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        dividend_data = dividend_data['historical']
        if len(dividend_data) != 0:
            dividend_data = pd.DataFrame(dividend_data)
            dividend_data = dividend_data.drop('label', axis=1)
            dividend_data = dividend_data.drop('adjDividend', axis=1)
            dividend_data = dividend_data.drop('recordDate', axis=1)
            dividend_data = dividend_data.drop('paymentDate', axis=1)
            dividend_data = dividend_data.drop('declarationDate', axis=1)
            dividend_data['date'] = dividend_data['date'].apply(lambda x: x.replace('-', ''))
            dividend_data = dividend_data.rename(columns={'date': 'endDate'})
            dividend_data['endDate'] = dividend_data['endDate'].apply(get_pre_quarter_end_date)
            dividend_data.drop_duplicates(subset=['endDate'], keep='first', inplace=True)
            dividend_data = dividend_data.reset_index(drop=True)
            dividend_data = pd.merge(enterprise_values_data, dividend_data, on=['endDate'], how='outer')
        else:
            dividend_data = enterprise_values_data
            dividend_data['dividend'] = 0.0
        dividend_data = dividend_data.fillna(0)
        dividend_data = dividend_data[dividend_data['symbol'] != 0]
        dividend_data = dividend_data.reset_index(drop=True)
        dividend_data.to_sql(dividend_exchange_name, con=engine, if_exists='append', index=False)
    print(dividend_exchange_name + ' data inited!')

def write_exchange_dividend_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    dividend_exchange_name = 'dividend_' + exchange.lower()
    # 创建元数据对象
    metadata = MetaData()
    if dividend_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(dividend_exchange_name + ' Table exists')
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        dividend_table = Table(dividend_exchange_name, metadata, autoload=True, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(dividend_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            fmp_symbol = stock_data['financialmodelingprep_symbol']
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
            endDate = last_date.replace('-', '')
            date_format = "%Y-%m-%d"
            last_date = datetime.datetime.strptime(last_date, date_format)
            last_date = last_date + datetime.timedelta(days=1)
            today = datetime.datetime.today()
            # 计算日期差
            delta = today - last_date
            quarter_num = int(float(delta.days/90))
            if quarter_num < 1:
                continue
            # 获得基本的企业数据
            try:
                url = 'https://financialmodelingprep.com/api/v3/enterprise-values/' + fmp_symbol + '?period=quarter&limit=' + str(quarter_num) + '&apikey=' + token
                enterprise_values_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(enterprise_values_data) == 0:
                continue
            enterprise_values_data = pd.DataFrame(enterprise_values_data)
            enterprise_values_data = enterprise_values_data[enterprise_values_data['date'] > endDate]
            if len(enterprise_values_data) == 0:
                continue
            enterprise_values_data = enterprise_values_data.reset_index(drop=True)
            enterprise_values_data['symbol'] = symbol
            enterprise_values_data['date'] = enterprise_values_data['date'].apply(lambda x: x.replace('-', ''))
            enterprise_values_data = enterprise_values_data.rename(columns={'date': 'endDate'})
            enterprise_values_data = enterprise_values_data.drop('stockPrice', axis=1)
            enterprise_values_data = enterprise_values_data.drop('marketCapitalization', axis=1)
            enterprise_values_data = enterprise_values_data.drop('minusCashAndCashEquivalents', axis=1)
            enterprise_values_data = enterprise_values_data.drop('addTotalDebt', axis=1)
            enterprise_values_data = enterprise_values_data.drop('enterpriseValue', axis=1)
            enterprise_values_data.drop_duplicates(subset=['endDate'], keep='first', inplace=True)
            enterprise_values_data = enterprise_values_data.reset_index(drop=True)
            # 获得分红数据
            try:
                url = 'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/' + fmp_symbol + '?period=quarter&limit=' + str(quarter_num+3) + '&apikey=' + token
                dividend_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            dividend_data = dividend_data['historical']
            if len(dividend_data) != 0:
                dividend_data = pd.DataFrame(dividend_data)
                dividend_data = dividend_data.drop('label', axis=1)
                dividend_data = dividend_data.drop('adjDividend', axis=1)
                dividend_data = dividend_data.drop('recordDate', axis=1)
                dividend_data = dividend_data.drop('paymentDate', axis=1)
                dividend_data = dividend_data.drop('declarationDate', axis=1)
                dividend_data['date'] = dividend_data['date'].apply(lambda x: x.replace('-', ''))
                dividend_data = dividend_data.rename(columns={'date': 'endDate'})
                dividend_data['endDate'] = dividend_data['endDate'].apply(get_pre_quarter_end_date)
                dividend_data.drop_duplicates(subset=['endDate'], keep='first', inplace=True)
                dividend_data = dividend_data.reset_index(drop=True)
                dividend_data = pd.merge(enterprise_values_data, dividend_data, on=['endDate'], how='outer')
            else:
                dividend_data = enterprise_values_data
                dividend_data['dividend'] = 0.0
            dividend_data = dividend_data.fillna(0)
            dividend_data = dividend_data[dividend_data['numberOfShares'] != 0]
            dividend_data = dividend_data.reset_index(drop=True)
            dividend_data = dividend_data[dividend_data['symbol'] != 0]
            dividend_data = dividend_data.reset_index(drop=True)
            for index, row in dividend_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(dividend_table.insert(), data)
                    print(symbol + ' ' + data['endDate'] + ' dividend_data inserted!')
                except:
                    pass
    else:
        print(dividend_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_dividend_data(exchange)

def write_daily_data_into_table(daily_exchange_name, stock_symbol_data):
    #建立连接
    metadata = MetaData()
    daily_table = Table(daily_exchange_name, metadata, autoload=True, autoload_with=engine)
    conn = engine.connect()
    for i in range(len(stock_symbol_data)):
        stock_data = stock_symbol_data.iloc[i]
        symbol = stock_data['symbol']
        print(daily_exchange_name + '_' + symbol + ':' + str(float(i)/len(stock_symbol_data)))
        fmp_symbol = stock_data['financialmodelingprep_symbol']
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
        last_date = datetime.datetime.strptime(last_date, date_format)
        last_date = last_date + datetime.timedelta(days=1)
        # 将下一天的日期转换回字符串格式
        last_date = last_date.strftime(date_format)
        today = datetime.datetime.today()
        today = today.strftime("%Y-%m-%d")
        try:
            url = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + fmp_symbol + '?from=' + last_date +'&to='+ str(today)+'&apikey='+token
            daily_data = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        insert_num = 0
        if len(daily_data) != 0:
            daily_data = daily_data['historical']
            daily_data = pd.DataFrame(daily_data)
            daily_data.insert(0, 'symbol', [symbol] * len(daily_data))
            # daily_data = daily_data.drop('adjClose', axis=1)
            daily_data = daily_data.drop('unadjustedVolume', axis=1)
            daily_data = daily_data.drop('change', axis=1)
            daily_data = daily_data.drop('changePercent', axis=1)
            daily_data = daily_data.drop('vwap', axis=1)
            daily_data = daily_data.drop('label', axis=1)
            daily_data = daily_data.drop('changeOverTime', axis=1)
            daily_data['date'] = daily_data['date'].apply(lambda x: x.replace('-', ''))
            daily_data.drop_duplicates(subset=['symbol', 'date'], keep='first', inplace=True)
            daily_data = daily_data.reset_index(drop=True)
            # daily_data.to_sql(daily_exchange_name, con=engine, if_exists='append', index=False)
            for index, row in daily_data.iterrows():
                data = row.to_dict()
                try:
                    conn.execute(daily_table.insert(), data)
                    insert_num += 1
                    # print(symbol + ' ' + data['date'] + ' daily_data inserted!')
                except:
                    pass
            print('insert_num:' + str(insert_num))
            # print('insert_num:' + str(len(daily_data)))

def init_exchange_daily_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    daily_exchange_name = 'daily_' + exchange
    if daily_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(daily_exchange_name + ' Table exists')
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
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()

        # num_processes = 2
        # processes = []
        #
        # for i in range(num_processes):
        #     process = multiprocessing.Process(target=write_daily_data_into_table, args=(daily_exchange_name,
        #                                                                                 stock_symbol_data[i]))
        #     processes.append(process)
        #
        # # 启动所有进程
        # for process in processes:
        #     process.start()
        #
        # # 等待所有进程完成
        # for process in processes:
        #     process.join()

        print(daily_exchange_name + ' data inited!')


def write_exchange_daily_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    daily_exchange_name = 'daily_' + exchange
    if daily_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(daily_exchange_name + ' Table exists')
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        stock_symbol_data = stock_symbol_data.sample(frac=1, random_state=None)
        stock_symbol_data = stock_symbol_data.reset_index(drop=True)
        #将stock_symbol_data一分为3，开线程写入daily_exchange_name
        split_num = int(len(stock_symbol_data)*0.333)
        stock_symbol_data_0 = stock_symbol_data[:split_num]
        stock_symbol_data_1 = stock_symbol_data[split_num:2*split_num]
        stock_symbol_data_2 = stock_symbol_data[2*split_num:]
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_0,))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_1,))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, stock_symbol_data_2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
        # num_processes = 2
        # processes = []
        #
        # for i in range(num_processes):
        #     process = multiprocessing.Process(target=write_daily_data_into_table, args=(daily_exchange_name,
        #                                                                                 stock_symbol_data[i]))
        #     processes.append(process)
        #
        # # 启动所有进程
        # for process in processes:
        #     process.start()
        #
        # # 等待所有进程完成
        # for process in processes:
        #     process.join()
    else:
        print(daily_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_daily_data(exchange)


def init_exchange_indicator_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    indicator_exchange_name = 'indicator_' + exchange.lower()
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
        income_table_name = 'income_' + exchange
        income_sql = "SELECT * FROM " + income_table_name
        income_data = pd.read_sql(income_sql, session.connection())
        balance_table_name = 'balance_' + exchange
        balance_sql = "SELECT * FROM " + balance_table_name
        balance_data = pd.read_sql(balance_sql, session.connection())
        cashflow_table_name = 'cashflow_' + exchange
        cashflow_sql = "SELECT * FROM " + cashflow_table_name
        cashflow_data = pd.read_sql(cashflow_sql, session.connection())
        dividend_table_name = 'dividend_' + exchange
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

def write_exchange_indicator_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    indicator_exchange_name = 'indicator_' + exchange.lower()
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
        income_table_name = 'income_' + exchange
        income_sql = "SELECT * FROM " + income_table_name
        income_data = pd.read_sql(income_sql, session.connection())
        balance_table_name = 'balance_' + exchange
        balance_sql = "SELECT * FROM " + balance_table_name
        balance_data = pd.read_sql(balance_sql, session.connection())
        cashflow_table_name = 'cashflow_' + exchange
        cashflow_sql = "SELECT * FROM " + cashflow_table_name
        cashflow_data = pd.read_sql(cashflow_sql, session.connection())
        dividend_table_name = 'dividend_' + exchange
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
        init_exchange_indicator_data(exchange)

def write_exchange_financial_data(exchange):
    t1 = threading.Thread(target=write_exchange_income_data, args=(exchange,))
    t2 = threading.Thread(target=write_exchange_balance_data, args=(exchange,))
    t3 = threading.Thread(target=write_exchange_cashflow_data, args=(exchange,))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

def write_exchange_dividend_and_daily_data(exchange):
    t1 = threading.Thread(target=write_exchange_dividend_data, args=(exchange,))
    t2 = threading.Thread(target=write_exchange_daily_data, args=(exchange,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

def write_exchange_data(exchange):
    write_exchange_stock_symbol_data(exchange)
    write_exchange_financial_data(exchange)
    write_exchange_dividend_and_daily_data(exchange)
    write_exchange_indicator_data(exchange)

def get_all_countries():
    url = 'https://financialmodelingprep.com/api/v3/get-all-countries?apikey=' + token
    countries = get_jsonparsed_data(url)
    return countries

def get_country_stock(country):
    url = 'https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&country=' + country + '&apikey=' + token
    stock = get_jsonparsed_data(url)
    exchange_list = []
    for i in range(len(stock)):
        if stock[i]['exchangeShortName'] not in exchange_list:
            exchange_list.append(stock[i]['exchangeShortName'])
    return exchange_list

def get_exchange_stock(exchange):
    url = 'https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&exchange=' + exchange + '&apikey='+token
    stock = get_jsonparsed_data(url)
    return stock

def find_union(A, B):
    A = set(A)
    B = set(B)
    union_set = A.union(B)
    return list(union_set)

def get_all_exchanges():
    exchange_list = []
    countries = get_all_countries()
    for i in tqdm(range(len(countries))):
        country = countries[i]
        country_exchange_list = get_country_stock(country)
        exchange_list = find_union(exchange_list, country_exchange_list)
        print(exchange_list)
    exchange_list_data = pd.DataFrame(data=exchange_list)
    exchange_list_data.to_csv('exchange_all.csv', index=False)

def delete_exchange_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    table_names = inspector.get_table_names()
    drop_list = []
    for i in range(len(table_names)):
        table_name = table_names[i]
        if exchange in table_name:
            drop_list.append(table_name)
    print(drop_list)
    metadata = MetaData()
    for i in range(len(drop_list)):
        drop_name = drop_list[i]
        drop_table = Table(drop_name, metadata, autoload=True, autoload_with=engine)
        drop_table.drop(engine)
        print(drop_name + ' deleted!')

def write_data():
    not_list = ['Swiss', 'Toronto', 'Johannesburg', 'Jakarta', 'Stockholm', 'Oslo', 'Tokyo', 'Saudi', 'Brussels', 'Arca',
                'SaoPaulo', 'Thailand', 'Canadian', 'Australian', 'Korea', 'Warsaw', 'HongKong']
    for key, value in exchange_reference_dict.items():
        exchange = key
        if exchange in not_list:
            pass
        else:
            print(exchange)
            write_exchange_data(exchange)

def get_candidate_exchanges():
    url = 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=' + token
    symbol_with_financial_data = get_jsonparsed_data(url)
    candidate_exchanges = []
    count_symbols_num = 0
    for i in tqdm(range(len(exchange_all))):
        exchange = exchange_all[i]
        stock_data = get_exchange_stock(exchange)
        if len(stock_data) < 200:
            continue
        else:
            exchange_symbol_list = []
            for j in range(len(stock_data)):
                stock = stock_data[j]
                symbol = stock['symbol']
                exchange_symbol_list.append(symbol)
            intersection_list = list(set(symbol_with_financial_data) & set(exchange_symbol_list))
            intersection_num = len(intersection_list)
            if intersection_num > 200:
                candidate_exchanges.append(exchange)
                count_symbols_num += intersection_num
                # print(exchange)
                print(candidate_exchanges)
                print(intersection_num)
        # print(stock_data)

def get_exchange_financial_counts(exchange):
    url = 'https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey=' + token
    symbol_with_financial_data = get_jsonparsed_data(url)
    stock_data = get_exchange_stock(exchange)
    exchange_symbol_list = []
    for j in range(len(stock_data)):
        stock = stock_data[j]
        symbol = stock['symbol']
        exchange_symbol_list.append(symbol)
    intersection_list = list(set(symbol_with_financial_data) & set(exchange_symbol_list))
    print(len(intersection_list))

if __name__ == "__main__":
    # get_country_stock('KR')
    # update_exchange_stock_symbol_data('Shenzhen')
    # get_exchange_stock_symbol_data('Shenzhen')
    # update_exchange_data('Shenzhen')
    # init_exchange_indicator_data('Shenzhen')
    # update_exchange_financial_data('Taiwan')
    # get_country_stock('CN')
    # pro = ts.pro_api()
    # df = pro.fina_indicator(ts_code='300989.SZ')
    # df.to_csv('300989_fina_indicator.csv')
    # print(df)
    # init_exchange_indicator_data('Taiwan')
    # update_exchange_data('Taiwan')
    # write_exchange_indicator_data('Seoul')
    # write_exchange_dividend_and_daily_data('HongKong')
    # write_exchange_data('Shenzhen')
    # write_exchange_dividend_and_daily_data('Shanghai')
    # write_exchange_indicator_data('Shanghai')
    # init_exchange_stock_symbol_data('Nasdaq')
    # write_data()
    # get_candidate_exchanges()
    # print(len(candidate_exchanges))
    # get_exchange_financial_counts('PNK')
    # write_data()
    # write_exchange_data('Pnk')
    write_exchange_data('Shanghai')
    write_exchange_data('Shenzhen')
    # write_exchange_indicator_data('Nasdaq')
    # write_exchange_dividend_and_daily_data('India')
    # get_candidate_exchanges()
    # stock_data = get_exchange_stock('PNK')
    # print(stock_data)