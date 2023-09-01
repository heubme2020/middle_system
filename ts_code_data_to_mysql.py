import tools
import time
import pymysql
import pandas as pd
import tushare as ts
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

def get_ts_code_data():
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data = ts_code_data_L.append(ts_code_data_D)
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = ts_code_data.append(ts_code_data_P)
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    return ts_code_data


def get_daily_data(start_date, end_date):
    date_list = tools.get_date_list(start_date, end_date)
    daily_data = pro.daily(trade_date = '19900101')
    for i in range(len(date_list)):
        daily_dataframe = pro.daily(trade_date = date_list[i])
        daily_data = pd.concat([daily_data, daily_dataframe])
        print(float(i)/float(len(date_list)))
    return daily_data

def get_weekly_data(start_date, end_date):
    date_list = tools.get_date_list(start_date, end_date)
    weekly_data = pro.weekly(trade_date = '19900101')
    for i in range(len(date_list)):
        weekly_dataframe = pro.weekly(trade_date = date_list[i])
        weekly_data = pd.concat([weekly_data, weekly_dataframe])
        print(float(i)/float(len(date_list)))
    return weekly_data

def get_monthly_data(start_date, end_date):
    date_list = tools.get_date_list(start_date, end_date)
    monthly_data = pro.monthly(trade_date = '19900101')
    for i in range(len(date_list)):
        monthly_dataframe = pro.monthly(trade_date = date_list[i])
        monthly_data = pd.concat([monthly_data, monthly_dataframe])
        print(float(i)/float(len(date_list)))
    return monthly_data

monthly_data = get_monthly_data('19900101', '20000101')

monthly_data.to_csv("monthly_1990_2000.csv")
exit()
# daily_data = get_daily_data('19900101', '20000101')
# daily_data.to_csv("daily_1990_2000.csv")

weekly_data = get_weekly_data('19900101', '20000101')

weekly_data.to_csv("weekly_1990_2000.csv")
exit()
daily_data = pd.read_csv('daily_1990_2000.csv')
open_data = daily_data['open']
print(open_data)
# plt.plot(open_data)
# plt.show()
open_data_box_cox, lambda0 = boxcox(open_data)
print(max(open_data_box_cox))
print(min(open_data_box_cox))
# plt.plot(open_data_box_cox)
# plt.show()
print(lambda0)
exit()
ts_code_data = get_ts_code_data()
