import tushare as ts
import time
import numpy as np
import tushare_tools as tt
import random
import joblib
import os
import tools
import pandas as pd
from joblib import dump, load
from tushare_tools import fina_indicator_param_list
from sklearn.preprocessing import StandardScaler
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

def get_middle_system_stock_code():
    today = tools.get_today()
    hour = time.strftime('%H', time.localtime())
    if int(hour) < 17:
        today = tools.get_delta_date(today, -2)
    print(today)
    candidate_list = []
    #获取今天在交易股票
    stock_list = pro.stock_basic(exchange='', list_status='L')

    #根据股息率ttm>0，挑选股票
    daily_basic_data = pro.daily_basic(trade_date=today)
    daily_basic_data = daily_basic_data.drop(daily_basic_data[daily_basic_data['dv_ratio'] < 0.0001].index)
    daily_basic_data = daily_basic_data.reset_index(drop=True)
    print(daily_basic_data)
    dv_list = daily_basic_data['ts_code']

    #根据毛利率>10%，營收爲正，挑选股票
    bak_basic_data = pro.bak_basic(trade_date=today)
    bak_basic_data = bak_basic_data.drop(bak_basic_data[bak_basic_data['gpr'] < 10].index)
    bak_basic_data = bak_basic_data.reset_index(drop=True)
    bak_basic_data = bak_basic_data.drop(bak_basic_data[bak_basic_data['rev_yoy'] < 0].index)
    bak_basic_data = bak_basic_data.reset_index(drop=True)
    print(bak_basic_data)
    gpr_list = bak_basic_data['ts_code']

    dv_gpr_list = list(set(dv_list)&set(gpr_list))
    print(dv_gpr_list)
    #根据q_roe>1%，int_to_talcap<50(带息债务/全部投入资本), 挑选股票,用上上个季度的数据
    start_date, end_date = tools.get_season_border(today, -2)
    fina_indicator_data = pro.fina_indicator_vip(start_date=start_date, end_date=end_date)
    fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['ts_code'], keep='first')
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    fina_indicator_data = fina_indicator_data.fillna(0.0)
    fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['q_roe'] < 1].index)
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['int_to_talcap'] > 50].index)
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    print(fina_indicator_data)
    roe_debt_list = fina_indicator_data['ts_code']
    dv_gpr_roe_debt_list = list(set(dv_gpr_list)&set(roe_debt_list))
    print(len(dv_gpr_roe_debt_list))
    print(dv_gpr_roe_debt_list)

    #同时进行估值判断，如果输出属于2类，则选择；
    #加载分类模型
    lgb = joblib.load('valuation_lgb_model.pkl')
    #加载标准化数据的参数
    scaler = load('valuation_std_scaler.bin')
    #初始化
    param_num = len(fina_indicator_param_list)
    columns_list = []
    for i in range(param_num * 3):
        columns_list.append(str(i))

    dv_gpr_roe_debt_valuation_list = []
    for i in range(len(dv_gpr_roe_debt_list)):
        print(float(i)/len(dv_gpr_roe_debt_list))
        try:
            ts_code = dv_gpr_roe_debt_list[i]
            start_date = tools.get_delta_date(today, -365)
            end_date = today
            data_list = []

            # 生成分类数据
            # income
            income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
            income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data = income_data.fillna(0.0)
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
            fina_indicator_data = fina_indicator_data.head(3)
            fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
            fina_indicator_data = fina_indicator_data.values
            fina_indicator_data0 = list(fina_indicator_data[0])
            fina_indicator_data1 = list(fina_indicator_data[0] - fina_indicator_data[1])
            fina_indicator_data2 = list(fina_indicator_data[0] + fina_indicator_data[2] - 2*fina_indicator_data[1])
            fina_indicator_data = fina_indicator_data0 + fina_indicator_data1 + fina_indicator_data2
            data_list = data_list + fina_indicator_data
            data = pd.DataFrame(data_list)
            data = data.T
            y_pred = lgb.predict(data)
            print(y_pred)
            if y_pred[0] == 2:
                print(ts_code)
                dv_gpr_roe_debt_valuation_list.append(ts_code)
        except:
            pass
    print(dv_gpr_roe_debt_valuation_list)
    print(len(dv_gpr_roe_debt_valuation_list))
    dv_gpr_roe_debt_valuation_data = pd.DataFrame(columns=['ts_code'], data=dv_gpr_roe_debt_valuation_list)
    dv_gpr_roe_debt_valuation_data.to_csv('dv_gpr_roe_debt_valuation_data.csv', mode='w')
    # dv_gpr_roe_debt_valuation_data = pd.read_csv('dv_gpr_roe_debt_valuation_data.csv')
    # print(dv_gpr_roe_debt_valuation_data)
    # dv_gpr_roe_debt_valuation_data = dv_gpr_roe_debt_valuation_data.drop(['Unnamed: 0'], axis='columns')
    # dv_gpr_roe_debt_valuation_data = list(dv_gpr_roe_debt_valuation_data['ts_code'])
    # print(dv_gpr_roe_debt_valuation_data)
    # stock_num = len(dv_gpr_roe_debt_valuation_data)
    # # #加载预测分类模型
    # predict_lgb = joblib.load('predict_lgb_model.pkl')
    # # #加载标准化数据的参数
    # predict_scaler = load('predict_std_scaler.bin')
    # trade_date = today
    # # 推薦的股票
    # candidate_list = []
    # for i in range(stock_num):
    #     print(float(i)/stock_num)
    #     try:
    #         ##开始生成数据
    #         data_list = []
    #         start_date = tools.get_delta_date(trade_date, -100)
    #         ts_code = dv_gpr_roe_debt_valuation_data[i]
    #         # daily
    #         daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         daily_data = daily_data.head(64)
    #         daily_data = daily_data.drop(columns=['ts_code', 'trade_date'])
    #         daily_data = daily_data.values
    #         daily_data = daily_data.reshape(1, -1)
    #         daily_data = list(daily_data[0])
    #         for j in range(64 * 9 - len(daily_data)):
    #             daily_data.append(0.0)
    #         data_list = data_list + daily_data
    #         # daily_basic
    #         daily_basic_data = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         daily_basic_data = daily_basic_data.head(64)
    #         daily_basic_data = daily_basic_data.drop(columns=['ts_code', 'trade_date'])
    #         daily_basic_data = daily_basic_data.values
    #         daily_basic_data = daily_basic_data.reshape(1, -1)
    #         daily_basic_data = list(daily_basic_data[0])
    #         for j in range(64 * 16 - len(daily_basic_data)):
    #             daily_basic_data.append(0.0)
    #         data_list = data_list + daily_basic_data
    #         # moneyflow
    #         moneyflow_data = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         moneyflow_data = moneyflow_data.head(64)
    #         moneyflow_data = moneyflow_data.drop(columns=['ts_code', 'trade_date'])
    #         moneyflow_data = moneyflow_data.values
    #         moneyflow_data = moneyflow_data.reshape(1, -1)
    #         moneyflow_data = list(moneyflow_data[0])
    #         for j in range(64 * 18 - len(moneyflow_data)):
    #             moneyflow_data.append(0.0)
    #         data_list = data_list + moneyflow_data
    #         # print(data_list)
    #         # print(ts_code)
    #         # print(trade_date)
    #         # stk_factor
    #         # stk_factor_data = pro.stk_factor(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         # stk_factor_data = stk_factor_data.head(64)
    #         # stk_factor_data = stk_factor_data.drop(columns=['ts_code', 'trade_date'])
    #         # stk_factor_data = stk_factor_data.values
    #         # stk_factor_data = stk_factor_data.reshape(1,-1)
    #         # stk_factor_data = list(stk_factor_data[0])
    #         # for i in range(64*33-len(stk_factor_data)):
    #         #     stk_factor_data.append(0.0)
    #         stk_factor_data = []
    #         for j in range(64 * 33):
    #             stk_factor_data.append(0.0)
    #         data_list = data_list + stk_factor_data
    #
    #         start_date = tools.get_delta_date(trade_date, -500)
    #         # weekly
    #         weekly_data = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         weekly_data = weekly_data.head(64)
    #         weekly_data = weekly_data.drop(columns=['ts_code', 'trade_date'])
    #         weekly_data = weekly_data.values
    #         weekly_data = weekly_data.reshape(1, -1)
    #         weekly_data = list(weekly_data[0])
    #         for j in range(64 * 9 - len(weekly_data)):
    #             weekly_data.append(0.0)
    #         data_list = data_list + weekly_data
    #
    #         start_date = tools.get_delta_date(trade_date, -2000)
    #         # monthly
    #         monthly_data = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         monthly_data = monthly_data.head(64)
    #         monthly_data = monthly_data.drop(columns=['ts_code', 'trade_date'])
    #         monthly_data = monthly_data.values
    #         monthly_data = monthly_data.reshape(1, -1)
    #         monthly_data = list(monthly_data[0])
    #         for j in range(64 * 9 - len(monthly_data)):
    #             monthly_data.append(0.0)
    #         data_list = data_list + monthly_data
    #
    #         # income
    #         income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
    #         income_data = income_data.reset_index(drop=True)
    #         income_data = income_data.fillna(0.0)
    #         income_data = income_data.head(16)
    #         income_data = income_data.drop(
    #             columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type',
    #                      'update_flag'])
    #         income_data = income_data.values
    #         income_data = income_data.reshape(1, -1)
    #         income_data = list(income_data[0])
    #         for j in range(16 * 77 - len(income_data)):
    #             income_data.append(0.0)
    #         data_list = data_list + income_data
    #         # balancesheet
    #         balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
    #         balancesheet_data = balancesheet_data.reset_index(drop=True)
    #         balancesheet_data = balancesheet_data.fillna(0.0)
    #         balancesheet_data = balancesheet_data.head(16)
    #         balancesheet_data = balancesheet_data.drop(
    #             columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type',
    #                      'update_flag'])
    #         balancesheet_data = balancesheet_data.values
    #         balancesheet_data = balancesheet_data.reshape(1, -1)
    #         balancesheet_data = list(balancesheet_data[0])
    #         for j in range(16 * 144 - len(balancesheet_data)):
    #             balancesheet_data.append(0.0)
    #         data_list = data_list + balancesheet_data
    #         # cashflow
    #         cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
    #         cashflow_data = cashflow_data.reset_index(drop=True)
    #         cashflow_data = cashflow_data.fillna(0.0)
    #         cashflow_data = cashflow_data.head(16)
    #         cashflow_data = cashflow_data.drop(
    #             columns=['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type',
    #                      'update_flag'])
    #         cashflow_data = cashflow_data.values
    #         cashflow_data = cashflow_data.reshape(1, -1)
    #         cashflow_data = list(cashflow_data[0])
    #         for j in range(16 * 89 - len(cashflow_data)):
    #             cashflow_data.append(0.0)
    #         # fina_indicator
    #         fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
    #         fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
    #         fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    #         fina_indicator_data = fina_indicator_data.fillna(0.0)
    #         fina_indicator_data = fina_indicator_data.head(16)
    #         fina_indicator_data = fina_indicator_data.drop(columns=['ts_code', 'ann_date', 'end_date'])
    #         fina_indicator_data = fina_indicator_data.values
    #         fina_indicator_data = fina_indicator_data.reshape(1, -1)
    #         fina_indicator_data = list(fina_indicator_data[0])
    #         for j in range(16 * 105 - len(fina_indicator_data)):
    #             fina_indicator_data.append(0.0)
    #         data_list = data_list + fina_indicator_data
    #         # print(len(data_list))
    #         # print(data_list)
    #         data = pd.DataFrame(data_list)
    #         data = data.T
    #         data.insert(loc=0, column='trade_date', value=trade_date)
    #         data.insert(loc=0, column='ts_code', value=ts_code)
    #
    #         # 进行预测
    #         predict_data = data
    #         predict_data = predict_data.drop('ts_code', axis='columns')
    #         predict_data = predict_data.drop('trade_date', axis='columns')
    #         predict_data = predict_scaler.transform(predict_data)
    #         y_pred = predict_lgb.predict(predict_data)
    #         print(y_pred)
    #         if y_pred[0] == 2:
    #             candidate_list.append(ts_code)
    #     except:
    #         pass
    # print(candidate_list)
    # candidate_data = pd.DataFrame(columns=['ts_code'], data=candidate_list)
    # candidate_data.to_csv('candidate_data.csv', mode='w')

if __name__ == "__main__":
    get_middle_system_stock_code()


