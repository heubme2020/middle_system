import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from growth_death_models import Decoder
import numpy as np
import pandas as pd
import random
import os
import cupy as cp
import csv
from dateutil.relativedelta import relativedelta
import datetime
import shutil
import math
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from catboost import CatBoostRegressor
from get_stock_data import get_exchange_data
from write_stock_data import write_exchange_data, get_country_stock, get_exchange_stock,  get_all_countries, exchange_reference_dict

not_standard_list = ['dividendRatio', 'pbInverse', 'debtToEquityRatio', 'quickRatio', 'ebitdaratio',
                     'operatingIncomeRatio','incomeBeforeTaxRatio', 'netIncomeRatio', 'grossProfitRatio',  'psInverse',
                     'cashFlowInverse', 'roe', 'dcfInverse', 'open', 'low', 'high', 'close', 'volume', 'marketCap',
                     'dayinyear', 'eps', 'epsdiluted', 'dividend', 'adjClose', 'debtInverse',
                     'totalStockholdersEquityInverse', 'quickRatioInverse', 'roeInverse', 'interestInverse',
                     'cashInverse', 'grossInverse', 'delta', 'turnoverRate', 'turnoverNetAssetRatio', 'dayoveryear']


def prepare_gbdt_data(folder):
    csv_name = 'growth_death_data_train.csv'
    if os.path.exists(csv_name):
        os.remove(csv_name)
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    random.shuffle(csv_files)
    for i in tqdm(range(len(csv_files))):
        csv_file = csv_files[i]
        data = pd.read_csv(csv_file)
        input_data = data.iloc[:-3, 2:4].join(data.iloc[:-3, 16:115])
        # input_data = data.iloc[:-4, 2:115]
        # print(input_data)
        input_data = input_data.T
        input_data = np.ravel(input_data.values)
        input_data = pd.DataFrame(input_data)
        fact_data = data.iloc[1, -2:]
        fact_data = np.ravel(fact_data.values)
        fact_data = pd.DataFrame(fact_data)
        data = pd.concat([input_data, fact_data], axis=0)
        data = data.reset_index()
        data.drop(columns=['index'], inplace=True)
        data = data.T
        if i == 0:
            data.to_csv(csv_name, mode='a', index=False)
        else:
            data.to_csv(csv_name, mode='a', header=False, index=False)

def train_catboost():
    train_data = pd.read_csv('growth_death_train_data_17.csv')
    # train_data = pd.read_csv('growth_death_train_data_17.csv', engine='pyarrow')
    print(len(train_data))
    train_data = train_data.dropna()
    train_data = train_data.sample(frac=1, random_state=None)
    train_data = train_data.reset_index(drop=True)
    print(len(train_data))

    train_num = len(train_data)
    ratio = 0.2
    validate_num = int(ratio * train_num)
    train_input = train_data.iloc[:-validate_num, 2:-2]
    train_growth = train_data.iloc[:-validate_num, -2]
    data_num = len(train_input)

    validate_input = train_data.iloc[-validate_num:, 2:-2]
    validate_growth = train_data.iloc[-validate_num:, -2]
    model_growth = CatBoostRegressor(iterations=data_num, learning_rate=0.01, min_data_in_leaf=23, depth=7,
                                     loss_function='MAPE', eval_metric='MAPE', task_type='GPU', devices='0:1', early_stopping_rounds=2000)
    # 训练模型，并设置验证集
    model_growth.fit(
        train_input, train_growth,
        eval_set=(validate_input, validate_growth),
        verbose=1
    )
    model_growth.save_model('catboost_growth_17.bin')

    #训练死亡模型
    train_data = train_data.sample(frac=1, random_state=None)
    train_data = train_data.dropna()
    train_data = train_data.reset_index(drop=True)
    print(len(train_data))

    train_num = len(train_data)
    ratio = 0.2
    validate_num = int(ratio * train_num)
    train_input = train_data.iloc[:-validate_num, 2:-2]
    train_death = train_data.iloc[:-validate_num, -1]
    data_num = len(train_input)

    validate_input = train_data.iloc[-validate_num:, 2:-2]
    validate_death = train_data.iloc[-validate_num:, -1]
    model_death = CatBoostRegressor(iterations=data_num, learning_rate=0.01, min_data_in_leaf=23, depth=7,
                                    loss_function='MAPE', eval_metric='MAPE', task_type='GPU', devices='0:1', early_stopping_rounds=2000)
    # 训练模型，并设置验证集
    model_death.fit(
        train_input, train_death,
        eval_set=(validate_input, validate_death),
        verbose=1
    )
    model_death.save_model('catboost_death_17.bin')


def find_nearest_thursday(input_date_str):
    # 将输入字符串日期转换为datetime对象
    input_date = datetime.datetime.strptime(input_date_str, '%Y%m%d')

    # 找到与输入日期最接近的上一个周四的日期
    weekday = input_date.weekday()
    days_until_thursday = (weekday - 3) % 7
    nearest_thursday = input_date - datetime.timedelta(days=days_until_thursday)

    # 返回最近的周四日期，以字符串格式输出
    return nearest_thursday.strftime('%Y%m%d')

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

def metric_forecast_financial_with_catboost_result(test_folder, model_path):
    device = torch.device(0)
    model = torch.load(model_path)
    test_data = pd.read_csv('growth_death_test_data_17.csv', engine='pyarrow')
    test_data = test_data.dropna()
    test_data = test_data.reset_index(drop=True)
    print(len(test_data))
    for i in range(len(exchange_list)):
        exchange = exchange_list[i]
        daily_data = pd.read_csv(exchange + '/daily_' + exchange + '.csv')
        if 'adjClose' in daily_data.columns:
            daily_data = daily_data.drop('adjClose', axis=1)
        income_data = pd.read_csv(exchange + '/income_' + exchange + '.csv')
        balance_data = pd.read_csv(exchange + '/balance_' + exchange + '.csv')
        cashflow_data = pd.read_csv(exchange + '/cashflow_' + exchange + '.csv')
        indicator_data = pd.read_csv(exchange + '/indicator_' + exchange + '.csv')
        mean_data = pd.read_csv(exchange + '/mean_' + exchange + '.csv')
        std_data = pd.read_csv(exchange + '/std_' + exchange + '.csv')
        # 合并财务相关数据
        financial_data = pd.merge(indicator_data, income_data, on=['symbol', 'endDate'], how='outer')
        financial_data = pd.merge(financial_data, balance_data, on=['symbol', 'endDate'], how='outer')
        financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
        financial_data = financial_data.dropna()
        financial_data = financial_data.reset_index(drop=True)
        # 删除股票数为0的行
        financial_data = financial_data[financial_data['numberOfShares'] != 0]
        financial_data = financial_data.reset_index(drop=True)
        # 删除totalStockholdersEquity为0的行
        financial_data = financial_data[financial_data['totalStockholdersEquity'] != 0]
        financial_data = financial_data.reset_index(drop=True)
        # 删除totalCurrentLiabilities为0的行
        financial_data = financial_data[financial_data['totalCurrentLiabilities'] != 0]
        financial_data = financial_data.reset_index(drop=True)

        #删除financial_data中endDate小于mean std data中最早的endDate的所有行
        financial_data = financial_data[financial_data['endDate'] >= mean_data['endDate'].iloc[0]]
        financial_data = financial_data.reset_index(drop=True)

        exchange_suffix = exchange_reference_dict[exchange][2]
        exchange_files = []
        for j in range(len(test_files)):
            test_file = test_files[j]
            if exchange_suffix in test_file:
                exchange_files.append(test_file)

        for j in range(len(exchange_files)):
            file = exchange_files[j]
            str_splits = file.split('_')
            date_split = str_splits[-1]
            symbol = str_splits[-2]
            symbols = symbol.split('\\')
            symbol = symbols[-1]
            date_splits = date_split.split('.')
            endDate = date_splits[-2]

            #截取这个日期节点的输入数据
            thursday = find_nearest_thursday(endDate)
            group = financial_data[financial_data['symbol'] == symbol]
            group = group.reset_index(drop=True)
            group = group[group['endDate'] < int(endDate)]
            group = group.reset_index(drop=True)
            sheet_count = len(group)
            if sheet_count < 17:
                continue
            group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
            for k in range(sheet_count - 1):
                dates = pd.date_range(start=group['endDate'].iloc[k], end=group['endDate'].iloc[k + 1])
                dates = dates.date.tolist()
                for k in range(1, len(dates) - 1):
                    date = dates[k]
                    date_str = date.strftime('%Y-%m-%d')
                    data_append = group.iloc[k].copy()
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
            # 合并daily信息
            daily_group = daily_data[daily_data['symbol'] == symbol]
            daily_group.reset_index(drop=True)
            daily_group = daily_data[daily_data['date'] <= int(thursday)]
            daily_group.reset_index(drop=True)
            group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
            group = group.dropna(axis=0, how='any')
            group.reset_index(inplace=True)
            group = group.drop(columns='index')
            # 添加总市值这一列
            group['marketCap'] = group['numberOfShares'] * group['close']
            # 添加分红率这一列
            group['dividendRatio'] = group['dividend'] / group['close']
            # 添加每股净资产换成市净率的倒数
            group['pbInverse'] = group['netAssetValuePerShare'] / group['close']
            # 添加每股销售额换成市销率的倒数
            group['psInverse'] = group['revenuePerShare'] / group['close']
            # 每股经营现金流换成对应的，我也不知道叫啥
            group['cashFlowInverse'] = group['cashFlowPerShare'] / group['close']
            # 每股的dcf换成对应的
            group['dcfInverse'] = group['dcfPerShare'] / group['close']
            # 处理其它几个价格
            group['open'] = group['open'] / group['close']
            group['low'] = group['low'] / group['close']
            group['high'] = group['high'] / group['close']
            # volume换成换手率%
            group['volume'] = group['volume'] * group['close'] / group['marketCap']
            # 加入date在一年中位置这一列
            group['dayinyear'] = group['date']
            group['dayinyear'] = group['dayinyear'].apply(lambda x: day_in_year(x))
            group['dayinyear'] = group['dayinyear'] / 365.0
            group = group.tail(768)
            group = group.reset_index(drop=True)
            col_names = group.columns.values
            # 单独处理下close和marketCap
            max_close = group['close'].max()
            min_close = group['close'].min()
            group['close'] = (group['close'] - min_close) / (max_close - min_close)
            max_marketCap = group['marketCap'].max()
            min_marketCap = group['marketCap'].min()
            group['marketCap'] = (group['marketCap'] - min_marketCap) / (max_marketCap - min_marketCap + 0.00001)
            # 获得上个的季度末日期
            date_str = datetime.datetime.strptime(str(thursday), '%Y%m%d').date()
            quarter_month = ((date_str.month - 1) // 3) * 3 + 1
            quarter_end_date = datetime.date(date_str.year, quarter_month, 1) + datetime.timedelta(days=-1)
            pre_endDate = int(quarter_end_date.strftime('%Y%m%d'))

            for k in range(2, len(group.columns)):
                col_name = col_names[k]
                # 如果不在not_standard_list列表，则进行归一化
                if col_name not in not_standard_list:
                    mean_value = mean_data.loc[mean_data['endDate'] == pre_endDate, col_name].item()
                    std_value = std_data.loc[std_data['endDate'] == pre_endDate, col_name].item()
                    group[col_name] = group[col_name] - mean_value
                    if std_value != 0:
                        group[col_name] = group[col_name] / std_value
                    else:
                        group[col_name] = 0
            input_data = group.iloc[:, 2:]
            input_data = input_data.values
            input_data = torch.tensor(input_data.astype(float), device=device)
            input_data = input_data.float()
            input_data = input_data.reshape(1, 768, 125)
            output_data = model(input_data)
            output_data = output_data.cpu()
            output_data = output_data.detach().numpy()
            output_data = pd.DataFrame(output_data[0, :, :])
            #
            file_data = pd.read_csv(file)
            file_data = file_data.head(17)
            fin_data = file_data.iloc[:, :4].join(file_data.iloc[:, 16:115])
            #获得季度末日期
            input_date = datetime.datetime.strptime(thursday, '%Y%m%d')
            quarter_end_date = input_date.replace(day=1, month=((input_date.month - 1) // 3 + 1) * 3, hour=23,
                                                  minute=59, second=59)
            quarter_end_date = quarter_end_date + datetime.timedelta(days=32)  # Add 32 days to handle edge cases
            quarter_end_date = quarter_end_date.replace(day=1) - datetime.timedelta(days=1)
            quarter_end_date = quarter_end_date.strftime('%Y%m%d')
            output_data.insert(loc=0, column='endDate', value=quarter_end_date)
            output_data.insert(loc=0, column='symbol', value=symbol)
            output_data.columns = fin_data.columns
            output_data = pd.concat([file_data, output_data], ignore_index=True)
            output_data = output_data.reset_index(drop=True)
            output_data.fillna(0, inplace=True)
            output_data.to_csv('output_data.csv', index=False)
            print(output_data)

def metric_catboost_model():
    growth_model = CatBoostRegressor(task_type='GPU')
    growth_model.load_model('catboost_growth_17.bin')
    death_model = CatBoostRegressor(task_type='GPU')
    death_model.load_model('catboost_death_17.bin')

    test_data = pd.read_csv('growth_death_test_data_17.csv', engine='pyarrow')
    test_data = test_data.dropna()
    test_data = test_data.reset_index(drop=True)
    growth_mae = 0
    death_mae = 0
    growth_rmse = 0
    death_rmse = 0
    growth_mape = 0
    death_mape = 0
    test_num = len(test_data)
    count_num = 0
    thresh = 1.17
    for i in tqdm(range(test_num)):
        input_data = test_data.iloc[i, 2:-2]
        fact_growth = test_data.iloc[i, -2]
        fact_death = test_data.iloc[i, -1]
        input_data = cp.asnumpy(input_data.T)
        predict_growth = growth_model.predict(input_data)
        predict_death = death_model.predict(input_data)
        if predict_growth/predict_death > thresh:
            growth_mae = growth_mae + abs(predict_growth - fact_growth)
            death_mae = death_mae + abs(predict_death - fact_death)
            growth_rmse = growth_rmse + (predict_growth - fact_growth)**2
            death_rmse = death_rmse + (predict_death - fact_death)**2
            growth_mape = growth_mape + abs((predict_growth - fact_growth)/fact_growth)
            death_mape = death_mape + abs((predict_death - fact_death)/fact_death)
            count_num = count_num + 1
    growth_mae = growth_mae/float(count_num)
    death_mae = death_mae/float(count_num)
    growth_rmse = math.sqrt(growth_rmse/float(count_num))
    death_rmse = math.sqrt(death_rmse/float(count_num))
    growth_mape = growth_mape/float(count_num)
    death_mape = death_mape/float(count_num)
    ratio = count_num/float(test_num)
    print('growth_mae:' + str(growth_mae))
    print('death_mae:' + str(death_mae))
    print('growth_rmse:' + str(growth_rmse))
    print('death_rmse:' + str(death_rmse))
    print('growth_mape:' + str(growth_mape))
    print('death_mape:' + str(death_mape))
    print('ratio:' + str(ratio))

def get_catboost_feature_importance():
    growth_model = CatBoostRegressor()
    growth_model.load_model('catboost_growth_17.bin')
    # 获取特征重要性得分
    growth_feature_importance = growth_model.get_feature_importance()
    growth_feature_importance = np.array(growth_feature_importance).reshape(101, 17)
    
    death_model = CatBoostRegressor()
    death_model.load_model('catboost_death_17.bin')
    # 获取特征重要性得分
    death_feature_importance = death_model.get_feature_importance()
    death_feature_importance = np.array(death_feature_importance).reshape(101, 17)

    feature_importance = growth_feature_importance + death_feature_importance
    feature_importance = feature_importance.T
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance = feature_importance.sum()
    feature_importance = pd.DataFrame(feature_importance)
    #删除其中的一部分
    feature_importance = feature_importance.T
    mean_data = pd.read_csv('mean_data.csv')
    feature_data = mean_data.iloc[:, 1:3].join(mean_data.iloc[:, 15:114])
    print(feature_data)
    feature_importance.columns = feature_data.columns
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(feature_importance)

def do_catboost():
    prepare_gbdt_data('growth_death_data/train')
    train_catboost()
    metric_catboost_model('growth_death_data/test')
    get_catboost_feature_importance()

def get_mean_std_data(data):
    data = data.drop('symbol', axis=1)
    data = data.drop('endDate', axis=1)
    col_names = data.columns.values
    mean_list = []
    std_list = []
    for k in range(len(col_names)):
        col_name = col_names[k]
        mean_value = data[col_name].mean()
        std_value = data[col_name].std()
        threshold = 3
        # 根据阈值筛选出异常值的索引
        outlier_indices = data.index[abs(data[col_name] - mean_value) > threshold * std_value]
        # print(outlier_indices)
        # 剔除包含异常值的行
        data_cleaned = data.drop(outlier_indices)
        data_cleaned = data_cleaned.reset_index(drop=True)
        # 计算剔除异常值后的均值和方差
        mean_cleaned = data_cleaned[col_name].mean()
        std_cleaned = data_cleaned[col_name].std()
        mean_list.append(mean_cleaned)
        std_list.append(std_cleaned)
    mean_data = pd.DataFrame([mean_list])
    std_data = pd.DataFrame([std_list])
    mean_data.columns = col_names
    std_data.columns = col_names
    return mean_data, std_data

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

def get_exchange_growth_death(exchange):
    get_exchange_data(exchange)
    growth_model = CatBoostRegressor(task_type='GPU')
    growth_model.load_model('catboost_growth_17.bin')
    death_model = CatBoostRegressor(task_type='GPU')
    death_model.load_model('catboost_death_17.bin')
    upper_exchange = exchange
    exchange = exchange.lower()
    income_data = pd.read_csv(upper_exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(upper_exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(upper_exchange + '/cashflow_' + exchange + '.csv')
    dividend_data = pd.read_csv(upper_exchange + '/dividend_' + exchange + '.csv')
    mean_data = pd.read_csv(exchange + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv(exchange + '/std_' + exchange + '.csv')

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
    financial_data = financial_data[financial_data['numberOfShares'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalStockholdersEquity为0的行
    financial_data = financial_data[financial_data['totalStockholdersEquity'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)
    #删除totalCurrentLiabilities为0的行
    financial_data = financial_data[financial_data['totalCurrentLiabilities'].astype(float) != 0]
    financial_data = financial_data.reset_index(drop=True)

    #predict结果
    predict_list = []
    groups = list(financial_data.groupby('symbol'))
    print(len(groups))
    #生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        symbol = groups[i][0]
        group = group.reset_index(drop=True)
        #目前必须上市17个季度后才可以预测
        if len(group) < 17:
            continue
        else:
            #下面进行归一化
            input_data = group.iloc[-17:]
            input_data = input_data.reset_index(drop=True)
            input_data = input_data.drop('symbol', axis=1)
            input_data = input_data.drop('endDate', axis=1)
            input_data = input_data.fillna(0)
            input_data.replace([np.inf, -np.inf], 0, inplace=True)
            col_names = input_data.columns.values
            for k in range(len(input_data.columns)):
                col_name = col_names[k]
                if col_name in not_standard_list:
                    continue
                mean_value = mean_data[col_name].iloc[-1]
                std_value = std_data[col_name].iloc[-1]
                input_data[col_name] = input_data[col_name] - mean_value
                if std_value != 0:
                    input_data[col_name] = input_data[col_name] / std_value
                else:
                    input_data[col_name] = 0
            input_data = input_data.T
            input_data = np.ravel(input_data.values)
            input_data = pd.DataFrame(input_data)
            input_data = cp.asnumpy(input_data.T)
            predict_growth = growth_model.predict(input_data)
            predict_growth = predict_growth[0]
            predict_death = death_model.predict(input_data)
            predict_death = predict_death[0]
            predict_result = {'symbol':symbol, 'growth':predict_growth, 'death':predict_death}
            predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_data['rank'] = predict_data['growth']/predict_data['death']
    predict_data = predict_data.sort_values('rank', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    predict_data.to_csv(exchange + '/grow_death_predict.csv', index=False)

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def get_growth_death():
    get_exchange_growth_death('Shenzhen')
    get_exchange_growth_death('Shanghai')
    predict_data_shenzhen = pd.read_csv('Shenzhen/grow_death_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv('Shanghai/grow_death_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='rank', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    predict_data = predict_data[predict_data['rank'] >= 1.17]
    pre_predict_data = pd.read_csv('candidate_symbols.csv')
    delete_symbols = pre_predict_data[~pre_predict_data['symbol'].isin(predict_data['symbol'])].dropna()
    add_symbols = predict_data[~predict_data['symbol'].isin(pre_predict_data['symbol'])].dropna()
    print('删除的股票：')
    print(delete_symbols)
    print('添加的股票：')
    print(add_symbols)
    predict_data.to_csv('candidate_symbols.csv', index=False)

def get_growth_death_of_a():
    get_exchange_growth_death('a')
    predict_data = pd.read_csv('a/grow_death_predict.csv', engine='pyarrow')
    predict_data.sort_values(by='rank', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    predict_data = predict_data[predict_data['rank'] >= 1.17]
    try:
        pre_predict_data = pd.read_csv('candidate_symbols.csv')
        delete_symbols = pre_predict_data[~pre_predict_data['symbol'].isin(predict_data['symbol'])].dropna()
        add_symbols = predict_data[~predict_data['symbol'].isin(pre_predict_data['symbol'])].dropna()
        print('删除的股票：')
        print(delete_symbols)
        print('添加的股票：')
        print(add_symbols)
    except:
        pass
    predict_data.to_csv('candidate_symbols.csv', index=False)

def get_symbol_rank(symbol):
    predict_data_shenzhen = pd.read_csv('Shenzhen/grow_death_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv('Shanghai/grow_death_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='rank', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    symbol_rank_data = predict_data[predict_data['symbol'] == symbol]
    print(symbol_rank_data)

if __name__ == '__main__':
    # get_exchange_growth_and_death_train_data('Kuala')
    # copy_growth_and_death_data(1)
    # metric_catboost_model('growth_death_data/test')
    # metric_torch_model('growth_death_data/test')
    # do_catboost()
    # prepare_gbdt_data('growth_death_data/train')
    # train_catboost()
    # metric_catboost_model()
    # get_catboost_feature_importance()
    # get_exchange_growth_death('Shanghai')
    # metric_forecast_financial_with_catboost_result('growth_death_data/test', 'transformerm_one_financial_forcast.pt')
    # thursday = find_nearest_thursday('20230630')
    # print(thursday)
    # get_growth_death()
    # get_country_stock('CR')
    # stocks = get_exchange_stock('SAU')
    # print(stocks)
    # get_all_countries()
    # get_growth_death()
    # get_symbol_rank('002461.SZSE')
    get_growth_death_of_a()
