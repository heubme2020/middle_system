import time

import numpy as np
import pandas as pd
from chinese_calendar import is_workday
import tools
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import datetime
import tushare as ts
import cv2
import os
import random

ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

##trade_param_list
trade_date_param_list = ['trade_date']
##存储财务指标数据的参数列表
fina_indicator_param_list = ['eps', 'dt_eps', 'total_revenue_ps', 'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps',
                             'undist_profit_ps', 'extra_item',
                             'profit_dedt', 'gross_margin', 'current_ratio', 'quick_ratio', 'cash_ratio', 'ar_turn',
                             'ca_turn', 'fa_turn',
                             'assets_turn', 'op_income', 'ebit', 'ebitda', 'fcff', 'fcfe', 'current_exint',
                             'noncurrent_exint',
                             'interestdebt', 'netdebt', 'tangible_asset', 'working_capital', 'networking_capital',
                             'invest_capital', 'retained_earnings', 'diluted2_eps',
                             'bps', 'ocfps', 'retainedps', 'cfps', 'ebit_ps', 'fcff_ps', 'fcfe_ps', 'netprofit_margin',
                             'grossprofit_margin', 'cogs_of_sales', 'expense_of_sales', 'profit_to_gr', 'saleexp_to_gr',
                             'adminexp_of_gr', 'finaexp_of_gr', 'impai_ttm',
                             'gc_of_gr', 'op_of_gr', 'ebit_of_gr', 'roe', 'roe_waa', 'roe_dt', 'roa', 'npta',
                             'roic', 'roe_yearly', 'roa2_yearly', 'assets_to_eqt', 'dp_assets_to_eqt', 'ca_to_assets',
                             'nca_to_assets', 'tbassets_to_totalassets',
                             'int_to_talcap', 'eqt_to_talcapital', 'currentdebt_to_debt', 'longdeb_to_debt',
                             'ocf_to_shortdebt', 'debt_to_eqt', 'eqt_to_debt', 'eqt_to_interestdebt',
                             'tangibleasset_to_debt', 'tangasset_to_intdebt', 'tangibleasset_to_netdebt', 'ocf_to_debt',
                             'turn_days', 'roa_yearly', 'roa_dp', 'fixed_assets',
                             'profit_to_op', 'q_saleexp_to_gr', 'q_gc_to_gr', 'q_roe', 'q_dt_roe', 'q_npta',
                             'q_ocf_to_sales', 'basic_eps_yoy',
                             'dt_eps_yoy', 'cfps_yoy', 'op_yoy', 'ebt_yoy', 'netprofit_yoy', 'dt_netprofit_yoy',
                             'ocf_yoy', 'roe_yoy',
                             'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'q_sales_yoy', 'q_op_qoq',
                             'equity_yoy']

def get_trade_date_list_period(start_date, end_date):
    trade_date_list = []
    trade_cal = pro.trade_cal(start_date=start_date, end_date=end_date)
    trade_cal = trade_cal.drop(trade_cal[trade_cal['is_open']==0].index)
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    for i in range(trade_cal.shape[0]):
        trade_date_list.append(trade_cal['cal_date'].iloc[i])
    return trade_date_list

def get_ts_code_list(trade_date):
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = pd.concat([ts_code_data_L, ts_code_data_D, ts_code_data_P])
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    ts_code_data = ts_code_data.reset_index(drop=True)
    ts_code_data = ts_code_data.drop(ts_code_data[ts_code_data['list_date']>trade_date].index)
    return ts_code_data['ts_code']

def get_trade_date_mean_std():
    # mean_list = [0.0]*len(trade_date_param_list)
    # mean_square_list = [0.0]*len(trade_date_param_list)
    # std_list = [0.0]*len(trade_date_param_list)
    # trade_date_mean_std = pd.DataFrame([mean_list, mean_square_list, std_list], columns=trade_date_param_list)
    # trade_date_mean_std.insert(trade_date_mean_std.shape[1], column='k', value=0.0)
    # trade_date_mean_std.to_csv('trade_date_mean_std.csv', index=None, mode='w')
    #随机统计数据
    date_today = tools.get_date_today()
    end_date = tools.get_delta_date(date_today, -183)
    trade_date_list = get_trade_date_list_period('19950101', end_date)
    while True:
        try:
            # 读取trade_date_mean_std
            trade_date_mean_std = pd.read_csv('trade_date_mean_std.csv')
            print(trade_date_mean_std)
            i = random.randint(0, len(trade_date_list))
            trade_date = trade_date_list[i]
            print(trade_date)
            k = trade_date_mean_std['k'][0]
            for j in range(len(trade_date_param_list)):
                param = trade_date_param_list[j]
                trade_date_mean_std[param][0] = (k*trade_date_mean_std[param][0] + int(trade_date))/(k + 1.0)
                trade_date_mean_std[param][1] = (k*trade_date_mean_std[param][1] + (int(trade_date))**2)/(k + 1.0)
                trade_date_mean_std[param][2] = ((trade_date_mean_std[param][1]*(k + 1.0) - ((trade_date_mean_std[param][0])**2)*(k + 1.0))/(k + 1.0))**0.5
            trade_date_mean_std['k'][0] = trade_date_mean_std['k'][0] + 1.0
            trade_date_mean_std['k'][1] = trade_date_mean_std['k'][1] + 1.0
            trade_date_mean_std['k'][2] = trade_date_mean_std['k'][2] + 1.0
            print(k)
            trade_date_mean_std.to_csv('trade_date_mean_std.csv', index=None, mode='w')
        except:
            pass
def get_fina_indicator_mean_std():
    # mean_list = [0.0]*len(fina_indicator_param_list)
    # mean_square_list = [0.0]*len(fina_indicator_param_list)
    # std_list = [0.0]*len(fina_indicator_param_list)
    # fina_indicator_mean_std = pd.DataFrame([mean_list, mean_square_list, std_list], columns=fina_indicator_param_list)
    # fina_indicator_mean_std.insert(fina_indicator_mean_std.shape[1], column='k', value=0.0)
    # fina_indicator_mean_std.to_csv('fina_indicator_mean_std.csv', index=None, mode='w')
    #随机统计数据
    date_today = tools.get_date_today()
    end_date = tools.get_delta_date(date_today, -183)
    trade_date_list = get_trade_date_list_period('19950101', end_date)
    while True:
        try:
            # 读取get_fina_indicator_mean_std
            fina_indicator_mean_std = pd.read_csv('fina_indicator_mean_std.csv')
            print(fina_indicator_mean_std)
            i = random.randint(0, len(trade_date_list))
            trade_date = trade_date_list[i]
            print(trade_date)
            ts_code_list = get_ts_code_list(trade_date)
            j = random.randint(0, len(ts_code_list))
            ts_code = ts_code_list[j]
            print(ts_code)
            start_date = tools.get_delta_date(trade_date, -183)
            end_date = tools.get_delta_date(trade_date, 183)
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            fina_indicator_data = fina_indicator_data.fillna(-1.0)
            m = random.randint(0, fina_indicator_data.shape[0])
            k = fina_indicator_mean_std['k'][0]
            for n in range(len(fina_indicator_param_list)):
                param = fina_indicator_param_list[n]
                fina_indicator_mean_std[param][0] = (k*fina_indicator_mean_std[param][0] + fina_indicator_data[param][m])/(k + 1.0)
                fina_indicator_mean_std[param][1] = (k*fina_indicator_mean_std[param][1] + fina_indicator_data[param][m]**2)/(k + 1.0)
                fina_indicator_mean_std[param][2] = ((fina_indicator_mean_std[param][1]*(k + 1.0) - ((fina_indicator_mean_std[param][0])**2)*(k + 1.0))/(k + 1.0))**0.5
            fina_indicator_mean_std['k'][0] = fina_indicator_mean_std['k'][0] + 1.0
            fina_indicator_mean_std['k'][1] = fina_indicator_mean_std['k'][1] + 1.0
            fina_indicator_mean_std['k'][2] = fina_indicator_mean_std['k'][2] + 1.0
            print(k)
            fina_indicator_mean_std.to_csv('fina_indicator_mean_std.csv', index=None, mode='w')
        except:
            pass
if __name__ == "__main__":
    # get_fina_indicator_mean_std()
    get_trade_date_mean_std()