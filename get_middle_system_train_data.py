import numpy as np
import tushare as ts
import time
import tushare_tools as tt
import random
import joblib
import os
import tools
import pandas as pd
from joblib import dump, load
from tushare_tools import fina_indicator_param_list
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

def get_middle_system_valuation_train_data():
    trade_date_list = tt.get_season_date_list_period('19950101', '20220930')
    ts_code_list = tt.get_ts_code_list_all()
    target_0_num = 0
    target_1_num = 0
    target_2_num = 0
    #生成预测模型的训练数据
    while True:
        try:
            i = random.randint(0, len(trade_date_list))
            j = random.randint(0, len(ts_code_list))
            trade_date = trade_date_list[i]
            ts_code = ts_code_list[j]
            # # 判断股息率
            # daily_basic_data = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            # if daily_basic_data.empty:
            #     continue
            # daily_basic_data = daily_basic_data.fillna(0.0)
            # if daily_basic_data['dv_ratio'][0] < 0.0001:
            #     continue
            # 判断毛利率和营收，這裏的數據一般是從2016年開始的，所以注釋掉
            # bak_basic_data = pro.bak_basic(ts_code=ts_code, trade_date=trade_date)
            # if bak_basic_data.empty:
            #     continue
            # bak_basic_data = bak_basic_data.fillna(0.0)
            # if bak_basic_data['gpr'][0] < 10:
            #     continue
            # if bak_basic_data['rev_yoy'][0] < 0:
            #     continue
            # # 判断roe和有息负债，用上上季度的数据，之所以用上上季度的數據，是因爲，預測的時候有可能上季度的季報還沒有出來，這樣爲了統一輸入
            # start_date, end_date = tools.get_season_border(trade_date, -2)
            # fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            # if fina_indicator_data.empty:
            #     continue
            # fina_indicator_data = fina_indicator_data.fillna(0.0)
            # if fina_indicator_data['q_roe'][0] < 1:
            #     continue
            # if fina_indicator_data['int_to_talcap'][0] > 50:
            #     continue
            ##开始生成数据
            data_list = []
            start_date = tools.get_delta_date(trade_date, -365)
            end_date = tools.get_delta_date(trade_date, 183)
            # income
            income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
            income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0.0)
            income_data = income_data.drop(income_data[income_data['end_date'] > trade_date].index)
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.head(3)

            income_data = income_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            income_data = income_data.values
            income_data0 = list(income_data[0])
            income_data1 = list(income_data[0] - income_data[1])
            income_data2 = list(income_data[0] + income_data[2] - 2*income_data[1])
            income_data = income_data0 + income_data1 + income_data2
            data_list = data_list + income_data
            # # balancesheet
            balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
            balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.fillna(0.0)
            balancesheet_data = balancesheet_data.drop(balancesheet_data[balancesheet_data['end_date'] > trade_date].index)
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.head(3)
            balancesheet_data = balancesheet_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            balancesheet_data = balancesheet_data.values
            balancesheet_data0 = list(balancesheet_data[0])
            balancesheet_data1 = list(balancesheet_data[0] - balancesheet_data[1])
            balancesheet_data2 = list(balancesheet_data[0] + balancesheet_data[2] - 2*balancesheet_data[1])
            balancesheet_data = balancesheet_data0 + balancesheet_data1 + balancesheet_data2
            data_list = data_list + balancesheet_data
            # # cashflow
            cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.fillna(0.0)
            cashflow_data = cashflow_data.drop(cashflow_data[cashflow_data['end_date'] > trade_date].index)
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.head(3)
            cashflow_data = cashflow_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            cashflow_data = cashflow_data.values
            cashflow_data0 = list(cashflow_data[0])
            cashflow_data1 = list(cashflow_data[0] - cashflow_data[1])
            cashflow_data2 = list(cashflow_data[0] + cashflow_data[2] - 2*cashflow_data[1])
            cashflow_data = cashflow_data0 + cashflow_data1 + cashflow_data2
            data_list = data_list + cashflow_data
            # # fina_indicator
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['end_date'] > trade_date].index)
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.head(3)
            fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data = fina_indicator_data.values
            fina_indicator_data0 = list(fina_indicator_data[0])
            fina_indicator_data1 = list(fina_indicator_data[0] - fina_indicator_data[1])
            fina_indicator_data2 = list(fina_indicator_data[0] + fina_indicator_data[2] - 2*fina_indicator_data[1])
            fina_indicator_data = fina_indicator_data0 + fina_indicator_data1 + fina_indicator_data2
            data_list = data_list + fina_indicator_data

            end_date = tools.get_delta_date(trade_date, -3)
            data = pd.DataFrame(data_list)
            data = data.T
            data.insert(loc=0, column='end_date', value=end_date)
            data.insert(loc=0, column='ts_code', value=ts_code)

            #判斷估值，根據股價
            pre_start_date = tools.get_delta_date(end_date, -183)
            pre_end_date = end_date
            daily_pre = pro.daily(ts_code=ts_code, start_date=pre_start_date, end_date=pre_end_date)
            pre_max_price = daily_pre['close'].max()
            next_start_date = end_date
            next_end_date = tools.get_delta_date(end_date, 183)
            daily_next = pro.daily(ts_code=ts_code, start_date=next_start_date, end_date=next_end_date)
            next_median_price = daily_next['close'].median()
            # next_max_price = daily_next['close'].max()
            # idmax = daily_next['close'].idxmax()
            # next_min_price = daily_next['close'].min()
            # idmin = daily_next['close'].idxmin()

            if next_median_price/pre_max_price > 1.2:
                data['target'] = 2
                target_2_num = target_2_num + 1
            elif next_median_price > pre_max_price:
                data['target'] = 1
                target_1_num = target_1_num + 1
                data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
            else:
                data['target'] = 0
                target_0_num = target_0_num + 1
            # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a')
            # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
            print(data)
            print('0:' + str(target_0_num))
            print('1:' + str(target_1_num))
            print('2:' + str(target_2_num))
        except:
            pass
def append_middle_system_valuation_train_data():
    today = tools.get_today()
    end_date = tools.get_delta_date(today, -365)
    trade_date_list = tt.get_season_date_list_period('19900101', end_date)
    ts_code_list = tt.get_ts_code_list_all()
    target_0_num = 0
    target_1_num = 0
    target_2_num = 0
    count = 0
    #加载分类模型
    lgb = joblib.load('valuation_lgb_model.pkl')
    # #加载标准化数据的参数
    # scaler = load('valuation_std_scaler.bin')
    #生成预测模型的训练数据
    while True:
    # for i in range(len(trade_date_list)):
        try:
            i = random.randint(0, len(trade_date_list))
            j = random.randint(0, len(ts_code_list))
            #     print(str(float(i)/len(trade_date_list)) + ':**************************************')
            trade_date = trade_date_list[i]
            # for j in range(len(ts_code_list)):
            #     print(float(j)/len(ts_code_list))
            ts_code = ts_code_list[j]
            # i = random.randint(0, len(trade_date_list))
            # j = random.randint(0, len(ts_code_list))
            # trade_date = trade_date_list[i]
            # ts_code = ts_code_list[j]
            ##因为样本数据太少，我们把所有前置条件都去掉
            # # 判断股息率
            # daily_basic_data = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            # if daily_basic_data.empty:
            #     continue
            # daily_basic_data = daily_basic_data.fillna(0.0)
            # if daily_basic_data['dv_ratio'][0] < 0.0001:
            #     continue
            # # 判断毛利率和营收，這裏的數據一般是從2016年開始的，所以注釋掉
            # # bak_basic_data = pro.bak_basic(ts_code=ts_code, trade_date=trade_date)
            # # if bak_basic_data.empty:
            # #     continue
            # # bak_basic_data = bak_basic_data.fillna(0.0)
            # # if bak_basic_data['gpr'][0] < 10:
            # #     continue
            # # if bak_basic_data['rev_yoy'][0] < 0:
            # #     continue
            # # 判断roe和有息负债，用上上季度的数据，之所以用上上季度的數據，是因爲，預測的時候有可能上季度的季報還沒有出來，這樣爲了統一輸入
            # start_date, end_date = tools.get_season_border(trade_date, -2)
            # fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            # if fina_indicator_data.empty:
            #     continue
            # fina_indicator_data = fina_indicator_data.fillna(0.0)
            # if fina_indicator_data['q_roe'][0] < 1:
            #     continue
            # if fina_indicator_data['int_to_talcap'][0] > 50:
            #     continue
            ##开始生成数据
            data_list = []
            start_date = tools.get_delta_date(trade_date, -365)
            end_date = tools.get_delta_date(trade_date, 183)
            # income
            income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
            income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0.0)
            income_data = income_data.drop(income_data[income_data['end_date'] > trade_date].index)
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.head(3)

            income_data = income_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            income_data = income_data.values
            income_data0 = list(income_data[0])
            income_data1 = list(income_data[0] - income_data[1])
            income_data2 = list(income_data[0] + income_data[2] - 2*income_data[1])
            income_data = income_data0 + income_data1 + income_data2
            data_list = data_list + income_data
            # # balancesheet
            balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
            balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.fillna(0.0)
            balancesheet_data = balancesheet_data.drop(balancesheet_data[balancesheet_data['end_date'] > trade_date].index)
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.head(3)
            balancesheet_data = balancesheet_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            balancesheet_data = balancesheet_data.values
            balancesheet_data0 = list(balancesheet_data[0])
            balancesheet_data1 = list(balancesheet_data[0] - balancesheet_data[1])
            balancesheet_data2 = list(balancesheet_data[0] + balancesheet_data[2] - 2*balancesheet_data[1])
            balancesheet_data = balancesheet_data0 + balancesheet_data1 + balancesheet_data2
            data_list = data_list + balancesheet_data
            # # cashflow
            cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.fillna(0.0)
            cashflow_data = cashflow_data.drop(cashflow_data[cashflow_data['end_date'] > trade_date].index)
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.head(3)
            cashflow_data = cashflow_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            cashflow_data = cashflow_data.values
            cashflow_data0 = list(cashflow_data[0])
            cashflow_data1 = list(cashflow_data[0] - cashflow_data[1])
            cashflow_data2 = list(cashflow_data[0] + cashflow_data[2] - 2*cashflow_data[1])
            cashflow_data = cashflow_data0 + cashflow_data1 + cashflow_data2
            data_list = data_list + cashflow_data
            # # fina_indicator
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['end_date'] > trade_date].index)
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.head(3)
            fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data = fina_indicator_data.values
            fina_indicator_data0 = list(fina_indicator_data[0])
            fina_indicator_data1 = list(fina_indicator_data[0] - fina_indicator_data[1])
            fina_indicator_data2 = list(fina_indicator_data[0] + fina_indicator_data[2] - 2*fina_indicator_data[1])
            fina_indicator_data = fina_indicator_data0 + fina_indicator_data1 + fina_indicator_data2
            data_list = data_list + fina_indicator_data

            end_date = tools.get_delta_date(trade_date, -3)
            data = pd.DataFrame(data_list)
            data = data.T
            y_pred = lgb.predict(data)
            # y_pred_prob = lgb.predict_proba(data)
            # print(y_pred_prob)

            data.insert(loc=0, column='end_date', value=end_date)
            data.insert(loc=0, column='ts_code', value=ts_code)

            #判斷估值，根據股價
            pre_start_date = tools.get_delta_date(end_date, -183)
            pre_end_date = end_date
            daily_pre = pro.daily(ts_code=ts_code, start_date=pre_start_date, end_date=pre_end_date)
            pre_max_price = daily_pre['close'].max()
            next_start_date = end_date
            next_end_date = tools.get_delta_date(end_date, 183)
            daily_next = pro.daily(ts_code=ts_code, start_date=next_start_date, end_date=next_end_date)
            next_median_price = daily_next['close'].median()

            if next_median_price/pre_max_price > 1.2:
                if y_pred[0] != 2:
                    data['target'] = 2
                    # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
                    target_2_num = target_2_num + 1
                    print(data)
                    print(y_pred)


            elif next_median_price > pre_max_price:
                if y_pred[0] != 1:
                    data['target'] = 1
                    target_1_num = target_1_num + 1
                    # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
                    print(data)
                    print(y_pred)
            else:
                if y_pred[0] != 0:
                    data['target'] = 0
                    target_0_num = target_0_num + 1
                    data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
                    print(data)
                    print(y_pred)
            # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a')
            # data.to_csv('middle_system_valuation_model_train_data.csv', index=False, mode='a', header=None)
            # print(data)
            count = count + 1
            print('0:' + str(target_0_num))
            print('1:' + str(target_1_num))
            print('2:' + str(target_2_num))
            print('count:' + str(count))
        except:
            pass
def get_middle_system_predict_train_data():
    trade_date_list = tt.get_trade_date_list_period('20011211', '20210101')
    ts_code_list = tt.get_ts_code_list_all()
    target_0_num = 0
    target_1_num = 0
    target_2_num = 0
    #生成预测模型的训练数据
    while True:
        try:
            i = random.randint(0, len(trade_date_list))
            j = random.randint(0, len(ts_code_list))
            trade_date = trade_date_list[i]
            ts_code = ts_code_list[j]
            #判断股息率
            daily_basic_data = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            if daily_basic_data.empty:
                continue
            daily_basic_data = daily_basic_data.fillna(0.0)
            if daily_basic_data['dv_ratio'][0] < 0.0001:
                continue
            # #判断毛利率和营收
            # bak_basic_data = pro.bak_basic(ts_code=ts_code, trade_date=trade_date)
            # if bak_basic_data.empty:
            #     continue
            # bak_basic_data = bak_basic_data.fillna(0.0)
            # if bak_basic_data['gpr'][0] < 10:
            #     continue
            # if bak_basic_data['rev_yoy'][0] < 0:
            #     continue
            #判断roe和有息负债，用上上季度的数据
            start_date, end_date = tools.get_season_border(trade_date, -2)
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if fina_indicator_data.empty:
                continue
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            if fina_indicator_data['q_roe'][0] < 1:
                continue
            if fina_indicator_data['int_to_talcap'][0] > 50:
                continue
            ##开始生成数据
            data_list = []
            start_date = tools.get_delta_date(trade_date, -100)
            #daily
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            daily_data = daily_data.head(64)
            daily_data = daily_data.drop(columns=['ts_code', 'trade_date'])
            daily_data = daily_data.values
            daily_data = daily_data.reshape(1,-1)
            daily_data = list(daily_data[0])
            for i in range(64*9-len(daily_data)):
                daily_data.append(0.0)
            data_list = data_list + daily_data
            #daily_basic
            daily_basic_data = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            daily_basic_data = daily_basic_data.head(64)
            daily_basic_data = daily_basic_data.drop(columns=['ts_code', 'trade_date'])
            daily_basic_data = daily_basic_data.values
            daily_basic_data = daily_basic_data.reshape(1,-1)
            daily_basic_data = list(daily_basic_data[0])
            for i in range(64*16-len(daily_basic_data)):
                daily_basic_data.append(0.0)
            data_list = data_list + daily_basic_data
            #moneyflow
            moneyflow_data = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            moneyflow_data = moneyflow_data.head(64)
            moneyflow_data = moneyflow_data.drop(columns=['ts_code', 'trade_date'])
            moneyflow_data = moneyflow_data.values
            moneyflow_data = moneyflow_data.reshape(1,-1)
            moneyflow_data = list(moneyflow_data[0])
            for i in range(64*18-len(moneyflow_data)):
                moneyflow_data.append(0.0)
            data_list = data_list + moneyflow_data
            #stk_factor
            stk_factor_data = pro.stk_factor(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            stk_factor_data = stk_factor_data.head(64)
            stk_factor_data = stk_factor_data.drop(columns=['ts_code', 'trade_date'])
            stk_factor_data = stk_factor_data.values
            stk_factor_data = stk_factor_data.reshape(1,-1)
            stk_factor_data = list(stk_factor_data[0])
            for i in range(64*33-len(stk_factor_data)):
                stk_factor_data.append(0.0)
            data_list = data_list + stk_factor_data

            start_date = tools.get_delta_date(trade_date, -500)
            #weekly
            weekly_data = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            weekly_data = weekly_data.head(64)
            weekly_data = weekly_data.drop(columns=['ts_code', 'trade_date'])
            weekly_data = weekly_data.values
            weekly_data = weekly_data.reshape(1,-1)
            weekly_data = list(weekly_data[0])
            for i in range(64*9-len(weekly_data)):
                weekly_data.append(0.0)
            data_list = data_list + weekly_data

            start_date = tools.get_delta_date(trade_date, -2000)
            #monthly
            monthly_data = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            monthly_data = monthly_data.head(64)
            monthly_data = monthly_data.drop(columns=['ts_code', 'trade_date'])
            monthly_data = monthly_data.values
            monthly_data = monthly_data.reshape(1,-1)
            monthly_data = list(monthly_data[0])
            for i in range(64*9-len(monthly_data)):
                monthly_data.append(0.0)
            data_list = data_list + monthly_data

            #income
            income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0.0)
            income_data = income_data.head(16)
            income_data = income_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            income_data = income_data.values
            income_data = income_data.reshape(1,-1)
            income_data = list(income_data[0])
            for i in range(16*77-len(income_data)):
                income_data.append(0.0)
            data_list = data_list + income_data
            #balancesheet
            balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.fillna(0.0)
            balancesheet_data = balancesheet_data.head(16)
            balancesheet_data = balancesheet_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            balancesheet_data = balancesheet_data.values
            balancesheet_data = balancesheet_data.reshape(1,-1)
            balancesheet_data = list(balancesheet_data[0])
            for i in range(16*144-len(balancesheet_data)):
                balancesheet_data.append(0.0)
            data_list = data_list + balancesheet_data
            #cashflow
            cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.fillna(0.0)
            cashflow_data = cashflow_data.head(16)
            cashflow_data = cashflow_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            cashflow_data = cashflow_data.values
            cashflow_data = cashflow_data.reshape(1,-1)
            cashflow_data = list(cashflow_data[0])
            for i in range(16*89-len(cashflow_data)):
                cashflow_data.append(0.0)
            #fina_indicator
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            fina_indicator_data = fina_indicator_data.head(16)
            fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data = fina_indicator_data.values
            fina_indicator_data = fina_indicator_data.reshape(1,-1)
            fina_indicator_data = list(fina_indicator_data[0])
            for i in range(16*105-len(fina_indicator_data)):
                fina_indicator_data.append(0.0)
            data_list = data_list + fina_indicator_data
            # print(len(data_list))
            # print(data_list)
            data = pd.DataFrame(data_list)
            data = data.T
            data.insert(loc=0, column='trade_date', value=trade_date)
            data.insert(loc=0, column='ts_code', value=ts_code)
            #获取下一个季度的fina_indicator,根据变化进行label
            start_date = tools.get_delta_date(trade_date, -365)
            end_date = tools.get_delta_date(trade_date, 365)
            fina_indicator_data_pre = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            fina_indicator_data_pre = fina_indicator_data_pre.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data_pre = fina_indicator_data_pre.reset_index(drop=True)
            fina_indicator_data_pre = fina_indicator_data_pre.fillna(0.0)
            fina_indicator_data_pre = fina_indicator_data_pre.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data_next = pro.fina_indicator(ts_code=ts_code, start_date=trade_date, end_date=end_date)
            fina_indicator_data_next = fina_indicator_data_next.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data_next = fina_indicator_data_next.sort_values(by='end_date', ascending=True)
            fina_indicator_data_next = fina_indicator_data_next.reset_index(drop=True)
            fina_indicator_data_next = fina_indicator_data_next.fillna(0.0)
            fina_indicator_data_next = fina_indicator_data_next.drop(columns=['ts_code', 'ann_date', 'end_date'])

            fina_indicator_data_delta = fina_indicator_data_next.loc[0]-fina_indicator_data_pre.loc[0]
            if (fina_indicator_data_delta['grossprofit_margin'] > 0) and (fina_indicator_data_delta['q_dt_roe'] > 0) and (fina_indicator_data_delta['q_sales_yoy'] > 0):
                data['target'] = 2
                target_2_num = target_2_num + 1
            elif fina_indicator_data_delta['q_sales_yoy'] > 0:
                data['target'] = 1
                target_1_num = target_1_num + 1
            else:
                data['target'] = 0
                target_0_num = target_0_num + 1
            # print(data)
            print('0:' + str(target_0_num))
            print('1:' + str(target_1_num))
            print('2:' + str(target_2_num))
            print('**********************************************************')
            data.to_csv('middle_system_predict_model_train_data.csv', index=False, mode='a', header=None)
        except:
            pass
def append_middle_system_predict_train_data():
    trade_date_list = tt.get_trade_date_list_period('20011211', '20220630')
    ts_code_list = tt.get_ts_code_list_all()
    target_0_num = 0
    target_1_num = 0
    target_2_num = 0
    # #加载预测分类模型
    predict_lgb = joblib.load('predict_lgb_model.pkl')
    # #加载标准化数据的参数
    predict_scaler = load('predict_std_scaler.bin')
    #生成数据进行判断
    count = 1
    while True:
        try:
            i = random.randint(0, len(trade_date_list))
            j = random.randint(0, len(ts_code_list))
            trade_date = trade_date_list[i]
            ts_code = ts_code_list[j]
            #判断股息率
            daily_basic_data = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            if daily_basic_data.empty:
                continue
            daily_basic_data = daily_basic_data.fillna(0.0)
            if daily_basic_data['dv_ratio'][0] < 0.0001:
                continue
            # #判断毛利率和营收
            # bak_basic_data = pro.bak_basic(ts_code=ts_code, trade_date=trade_date)
            # if bak_basic_data.empty:
            #     continue
            # bak_basic_data = bak_basic_data.fillna(0.0)
            # if bak_basic_data['gpr'][0] < 10:
            #     continue
            # if bak_basic_data['rev_yoy'][0] < 0:
            #     continue
            #判断roe和有息负债，用上上季度的数据
            start_date, end_date = tools.get_season_border(trade_date, -2)
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if fina_indicator_data.empty:
                continue
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            if fina_indicator_data['q_roe'][0] < 1:
                continue
            if fina_indicator_data['int_to_talcap'][0] > 50:
                continue
            ##开始生成数据
            data_list = []
            start_date = tools.get_delta_date(trade_date, -100)
            #daily
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            daily_data = daily_data.head(64)
            daily_data = daily_data.drop(columns=['ts_code', 'trade_date'])
            daily_data = daily_data.values
            daily_data = daily_data.reshape(1,-1)
            daily_data = list(daily_data[0])
            for i in range(64*9-len(daily_data)):
                daily_data.append(0.0)
            data_list = data_list + daily_data
            #daily_basic
            daily_basic_data = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            daily_basic_data = daily_basic_data.head(64)
            daily_basic_data = daily_basic_data.drop(columns=['ts_code', 'trade_date'])
            daily_basic_data = daily_basic_data.values
            daily_basic_data = daily_basic_data.reshape(1,-1)
            daily_basic_data = list(daily_basic_data[0])
            for i in range(64*16-len(daily_basic_data)):
                daily_basic_data.append(0.0)
            data_list = data_list + daily_basic_data
            #moneyflow
            moneyflow_data = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            moneyflow_data = moneyflow_data.head(64)
            moneyflow_data = moneyflow_data.drop(columns=['ts_code', 'trade_date'])
            moneyflow_data = moneyflow_data.values
            moneyflow_data = moneyflow_data.reshape(1,-1)
            moneyflow_data = list(moneyflow_data[0])
            for i in range(64*18-len(moneyflow_data)):
                moneyflow_data.append(0.0)
            data_list = data_list + moneyflow_data
            # print(data_list)
            # print(ts_code)
            # print(trade_date)
            #stk_factor
            stk_factor_data = []
            stk_factor_data = pro.stk_factor(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            stk_factor_data = stk_factor_data.head(64)
            stk_factor_data = stk_factor_data.drop(columns=['ts_code', 'trade_date'])
            stk_factor_data = stk_factor_data.values
            stk_factor_data = stk_factor_data.reshape(1,-1)
            stk_factor_data = list(stk_factor_data[0])
            for i in range(64*33-len(stk_factor_data)):
                stk_factor_data.append(0.0)

            # for i in range(64*33):
            #     stk_factor_data.append(0.0)
            data_list = data_list + stk_factor_data

            start_date = tools.get_delta_date(trade_date, -500)
            #weekly
            weekly_data = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            weekly_data = weekly_data.head(64)
            weekly_data = weekly_data.drop(columns=['ts_code', 'trade_date'])
            weekly_data = weekly_data.values
            weekly_data = weekly_data.reshape(1,-1)
            weekly_data = list(weekly_data[0])
            for i in range(64*9-len(weekly_data)):
                weekly_data.append(0.0)
            data_list = data_list + weekly_data

            start_date = tools.get_delta_date(trade_date, -2000)
            #monthly
            monthly_data = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            monthly_data = monthly_data.head(64)
            monthly_data = monthly_data.drop(columns=['ts_code', 'trade_date'])
            monthly_data = monthly_data.values
            monthly_data = monthly_data.reshape(1,-1)
            monthly_data = list(monthly_data[0])
            for i in range(64*9-len(monthly_data)):
                monthly_data.append(0.0)
            data_list = data_list + monthly_data

            #income
            income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0.0)
            income_data = income_data.head(16)
            income_data = income_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            income_data = income_data.values
            income_data = income_data.reshape(1,-1)
            income_data = list(income_data[0])
            for i in range(16*77-len(income_data)):
                income_data.append(0.0)
            data_list = data_list + income_data
            #balancesheet
            balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
            balancesheet_data = balancesheet_data.reset_index(drop=True)
            balancesheet_data = balancesheet_data.fillna(0.0)
            balancesheet_data = balancesheet_data.head(16)
            balancesheet_data = balancesheet_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            balancesheet_data = balancesheet_data.values
            balancesheet_data = balancesheet_data.reshape(1,-1)
            balancesheet_data = list(balancesheet_data[0])
            for i in range(16*144-len(balancesheet_data)):
                balancesheet_data.append(0.0)
            data_list = data_list + balancesheet_data
            #cashflow
            cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            cashflow_data = cashflow_data.fillna(0.0)
            cashflow_data = cashflow_data.head(16)
            cashflow_data = cashflow_data.drop(columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'])
            cashflow_data = cashflow_data.values
            cashflow_data = cashflow_data.reshape(1,-1)
            cashflow_data = list(cashflow_data[0])
            for i in range(16*89-len(cashflow_data)):
                cashflow_data.append(0.0)
            #fina_indicator
            fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data = fina_indicator_data.reset_index(drop=True)
            fina_indicator_data = fina_indicator_data.fillna(0.0)
            fina_indicator_data = fina_indicator_data.head(16)
            fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data = fina_indicator_data.values
            fina_indicator_data = fina_indicator_data.reshape(1,-1)
            fina_indicator_data = list(fina_indicator_data[0])
            for i in range(16*105-len(fina_indicator_data)):
                fina_indicator_data.append(0.0)
            data_list = data_list + fina_indicator_data
            # print(len(data_list))
            # print(data_list)
            data = pd.DataFrame(data_list)
            data = data.T
            data.insert(loc=0, column='trade_date', value=trade_date)
            data.insert(loc=0, column='ts_code', value=ts_code)
            #获取下一个季度的fina_indicator,根据变化进行label
            start_date = tools.get_delta_date(trade_date, -365)
            end_date = tools.get_delta_date(trade_date, 365)
            fina_indicator_data_pre = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
            fina_indicator_data_pre = fina_indicator_data_pre.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data_pre = fina_indicator_data_pre.reset_index(drop=True)
            fina_indicator_data_pre = fina_indicator_data_pre.fillna(0.0)
            fina_indicator_data_pre = fina_indicator_data_pre.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data_next = pro.fina_indicator(ts_code=ts_code, start_date=trade_date, end_date=end_date)
            fina_indicator_data_next = fina_indicator_data_next.drop_duplicates(subset=['end_date'], keep='first')
            fina_indicator_data_next = fina_indicator_data_next.sort_values(by='end_date', ascending=True)
            fina_indicator_data_next = fina_indicator_data_next.reset_index(drop=True)
            fina_indicator_data_next = fina_indicator_data_next.fillna(0.0)
            fina_indicator_data_next = fina_indicator_data_next.drop(columns=['ts_code', 'ann_date', 'end_date'])

            fina_indicator_data_delta = fina_indicator_data_next.loc[0]-fina_indicator_data_pre.loc[0]
            #进行预测
            predict_data = data
            predict_data = predict_data.drop('ts_code', axis='columns')
            predict_data = predict_data.drop('trade_date', axis='columns')
            predict_data = predict_scaler.transform(predict_data)
            y_pred = predict_lgb.predict(predict_data)
            if (fina_indicator_data_delta['grossprofit_margin'] > 0) and (fina_indicator_data_delta['q_dt_roe'] > 0) and (fina_indicator_data_delta['q_sales_yoy'] > 0):
                if y_pred[0] != 2:
                    print(ts_code)
                    print(y_pred)
                    print('2*******************')
                    data['target'] = 2
                    target_2_num = target_2_num + 1
                    data.to_csv('middle_system_predict_model_train_data.csv', index=False, mode='a', header=None)
            elif fina_indicator_data_delta['q_sales_yoy'] > 0:
                if y_pred[0] != 1:
                    print(ts_code)
                    print(y_pred)
                    print('1*******************')
                    data['target'] = 1
                    target_1_num = target_1_num + 1
                    data.to_csv('middle_system_predict_model_train_data.csv', index=False, mode='a', header=None)
            else:
                if y_pred[0] != 0:
                    print(ts_code)
                    print(y_pred)
                    print('0*******************')
                    data['target'] = 0
                    target_0_num = target_0_num + 1
                    data.to_csv('middle_system_predict_model_train_data.csv', index=False, mode='a', header=None)
            # print(data)
            print('0:' + str(target_0_num))
            print('1:' + str(target_1_num))
            print('2:' + str(target_2_num))
            print('count:' + str(count))
            print('**********************************************************')
            count = count + 1
        except:
            pass
def train_valuation_lgb():
    data = pd.read_csv('middle_system_valuation_model_train_data.csv')
    today = tools.get_today()
    trade_date = tools.get_delta_date(today, -365)
    data = data.drop(data[data['end_date'] > int(trade_date)].index)
    data = data.reset_index(drop=True)
    print(data)
    # #删除重复行
    data = data.drop_duplicates(['ts_code', 'end_date', 'target'], keep='first')
    data = data.reset_index(drop=True)
    print(data)
    ##填充NaN
    data = data.fillna(0.0)
    print(data)
    # #删除有nan的行
    # data = data.dropna(axis=0, how='any')
    # data = data.reset_index(drop=True)
    # print(data)
    data = data.drop('ts_code', axis='columns')
    data = data.reset_index(drop=True)
    data = data.drop('end_date', axis='columns')
    data = data.reset_index(drop=True)
    data = shuffle(data)
    data = data.reset_index(drop=True)
    print(data)

    y = data.target
    x = data.drop('target', axis='columns')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    # dump(scaler, 'valuation_std_scaler.bin', compress=True)
    ## 定义 LightGBM 模型
    clf = LGBMClassifier(feature_fraction=0.8, learning_rate=0.1, max_depth=-1, num_leaves=64)
    # 在训练集上训练LightGBM模型
    clf.fit(x_train, y_train)
    # 模型存储
    joblib.dump(clf, 'valuation_lgb_model.pkl')
    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    booster = clf.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    print(feature_importance)
    feature_importance.to_csv('valuation_feature_importance.csv', index=False)


    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    # # 定义参数取值范围
    # learning_rate = [0.1, 0.3, 0.6]
    # feature_fraction = [0.5, 0.8, 1]
    # num_leaves = [16, 32, 64]
    # max_depth = [-1, 3, 5, 8]
    #
    # parameters = {'learning_rate': learning_rate,
    #               'feature_fraction': feature_fraction,
    #               'num_leaves': num_leaves,
    #               'max_depth': max_depth}
    # model = LGBMClassifier(n_estimators=50)
    #
    # ## 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)
    # clf = clf.fit(x_train, y_train)
    # print(clf.best_params_)
def train_predict_lgb():
    data = pd.read_csv('middle_system_predict_model_train_data.csv')
    data = data.drop('ts_code', axis='columns')
    data = data.reset_index(drop=True)
    data = data.drop('trade_date', axis='columns')
    data = data.reset_index(drop=True)
    data = shuffle(data)
    data = data.reset_index(drop=True)
    print(data)
    y = data.target
    x = data.drop('target', axis='columns')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    print(x_train)
    x_test = scaler.transform(x_test)
    dump(scaler, 'predict_std_scaler.bin', compress=True)
    ## 定义 LightGBM 模型
    clf = LGBMClassifier(feature_fraction=0.5, learning_rate=0.3, max_depth=-1, num_leaves=64)
    # 在训练集上训练LightGBM模型
    clf.fit(x_train, y_train)
    # 模型存储
    joblib.dump(clf, 'predict_lgb_model.pkl')
    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    booster = clf.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()
    # for (feature_name,importance) in zip(feature_name,importance):
    #     print (feature_name,importance)
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance.to_csv('feature_importance.csv', index=False)


    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    # # 定义参数取值范围
    # learning_rate = [0.1, 0.3, 0.6]
    # feature_fraction = [0.5, 0.8, 1]
    # num_leaves = [16, 32, 64]
    # max_depth = [-1, 3, 5, 8]
    #
    # parameters = {'learning_rate': learning_rate,
    #               'feature_fraction': feature_fraction,
    #               'num_leaves': num_leaves,
    #               'max_depth': max_depth}
    # model = LGBMClassifier(n_estimators=50)

    # ## 进行网格搜索
    # clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)
    # clf = clf.fit(x_train, y_train)
    # print(clf.best_params_)

if __name__ == "__main__":
    append_middle_system_valuation_train_data()
    # train_valuation_lgb()
    # get_middle_system_valuation_train_data()
    # stk_factor_data = pro.stk_factor(ts_code='002116.SZ', start_date='20220401', end_date='20220523')
    # print(stk_factor_data)
    # append_middle_system_train_data()
    # train_predict_lgb()
    # get_middle_system_train_data()
    # data = pd.read_csv('middle_system_predict_model_train_data.csv')
    # data_a = data.loc[:, 'ts_code':'trade_date']
    # data_b = data.loc[:, 'target']
    # data = data.drop(['ts_code', 'trade_date', 'target'], axis='columns')
    # # print(data.columns)
    # # for i in range(11232):
    # #     data.columns(i) = str(i)
    # idx = 0
    # for column in data:
    #     data.rename(columns={column: str(idx)}, inplace=True)
    #     idx = idx+1
    # # data.rename(columns={'300727.SZ': 'ts_code', '20191212': 'trade_date', '0': 'target'}, inplace=True)
    # # data = data.drop(data[data['0'] == 0].index)
    # # # data = data.drop(data[data['0'] == 1].index)
    #
    # data = pd.concat([data_a, data, data_b], axis=1)
    # data = shuffle(data)
    # print(data_a)
    # # print(data_b)
    # data = data.reset_index(drop=True)
    # print(data)
    #
    # data.to_csv('middle_system_predict_model_train_data.csv', mode='w', index=False)


