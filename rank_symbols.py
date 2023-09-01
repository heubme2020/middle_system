from write_stock_data import write_exchange_data, update_exchange_data
from get_stock_data import get_exchange_data
import datetime
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from dateutil.relativedelta import relativedelta
import torch

not_standard_list = ['dividendRatio', 'pbInverse', 'debtToEquityRatio', 'quickRatio', 'ebitdaratio',
                     'operatingIncomeRatio','incomeBeforeTaxRatio', 'netIncomeRatio', 'grossProfitRatio',  'psInverse',
                     'cashFlowInverse', 'roe', 'dcfInverse', 'open', 'low', 'high', 'close', 'volume', 'marketCap',
                     'dayinyear']

def get_pre_thursday(date):
    date = datetime.datetime.strptime(date, "%Y%m%d")
    current_weekday = date.weekday()
    days_to_subtract = (current_weekday - 3) % 7  # 计算与上一个周四相差的天数
    thursday = date - datetime.timedelta(days=days_to_subtract)
    thursday = thursday.strftime("%Y%m%d")
    return thursday

def get_pre_endDate(date):
    quarter_month = ((date.month - 1) // 3) * 3 + 1
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    endDate = quarter_end_date.strftime('%Y%m%d')
    return endDate

def get_next_quarter_end_dates(date, num_dates=4):
    # 将输入日期字符串转换为日期对象
    date = pd.to_datetime(str(date), format="%Y%m%d").date()
    # 获取指定数量的下一个季度末日期
    quarter_end_dates = pd.date_range(start=date, periods=num_dates, freq='Q-DEC').strftime("%Y%m%d").tolist()
    return quarter_end_dates

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
            threshold = 3
            # 根据阈值筛选出异常值的索引
            outlier_indices = filtered_data.index[abs(filtered_data[col_name] - mean_value) > threshold * std_value]
            # print(outlier_indices)
            # 剔除包含异常值的行
            filtered_data_cleaned = filtered_data.drop(outlier_indices)
            filtered_data_cleaned = filtered_data_cleaned.reset_index()
            filtered_data_cleaned.drop(columns=['index'], inplace=True)
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

def day_in_year(date):
    date = str(date)
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    date = datetime.date(year, month, day)
    idx = int(date.strftime('%j'))
    return idx

def date_back(date_str):
    date_str = str(date_str)
    str_list = list(date_str)  # 字符串转list
    str_list.insert(4, '-')  # 在指定位置插入字符串
    str_list.insert(7, '-')  # 在指定位置插入字符串
    str_out = ''.join(str_list)  # 空字符连接
    return str_out

def rank_exchange_symbols(exchange):
    days_input = 768
    days_output = 256
    #加载财务预测模型
    device = torch.device(0)
    model = torch.load('decoder_financial_forcast.pt')
    print(model.eval())

    # #更新数据
    # today = datetime.date.today()
    # weekday = today.weekday()
    # if weekday < 5:
    #     update_exchange_data(exchange)
    # else:
    #     write_exchange_data(exchange)
    # get_exchange_data(exchange)

    #生成财务预测数据
    daily_data = pd.read_csv(exchange + '/daily_' + exchange + '.csv')
    income_data = pd.read_csv(exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(exchange + '/cashflow_' + exchange + '.csv')
    indicator_data = pd.read_csv(exchange + '/indicator_' + exchange + '.csv')
    #合并财务相关数据
    financial_data = pd.merge(indicator_data, income_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    financial_data = financial_data.dropna()
    financial_data = financial_data.reset_index(drop=True)
    #删除股票数为0的行
    financial_data = financial_data[financial_data['numberOfShares'] != 0]
    financial_data = financial_data.reset_index(drop=True)
    #先获得离今天最近的pre_endDate
    today = datetime.date.today()
    pre_endDate = get_pre_endDate(today)
    groups = list(financial_data.groupby('symbol'))
    # #生成对应的均值，方差矩阵
    mean_data, std_data = get_mean_std_data(financial_data)
    #开始遍历股票补全数据到financial_data，一般不会超过3个季度数据没有更新
    for j in range(len(groups)):
        group = groups[j][1]
        group = group.sort_values('endDate')
        group = group.reset_index(drop=True)
        #如果财务信息季度数小于13直接跳过
        if len(group) < 13:
            continue
        last_endDate = group['endDate'].iloc[-1]
        #如果不到pre_endDate则进行补全
        if int(last_endDate) < int(pre_endDate):
            group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
            for k in range(len(group)-1):
                dates = pd.date_range(start=group['endDate'].iloc[k], end=group['endDate'].iloc[k + 1])
                dates = dates.date.tolist()
                for m in range(1, len(dates) - 1):
                    date = dates[m]
                    date_str = date.strftime('%Y-%m-%d')
                    data_append = group.iloc[k].copy()
                    data_append['endDate'] = date_str
                    group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
            last_endDate = date_back(str(last_endDate))
            date = datetime.datetime.strptime(last_endDate, '%Y-%m-%d')
            next_quarter_end = date + relativedelta(months=3)
            next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
            dates = pd.date_range(start=last_endDate, end=next_quarter_end_str)
            dates = dates.date.tolist()
            for k in range(1, len(dates) - 1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = group.iloc[-1].copy()
                data_append['endDate'] = date_str
                group = pd.concat([group, data_append.to_frame().T], ignore_index=True)
            group['endDate'] = group['endDate'].apply(lambda x: x.replace('-', ''))
            group.sort_values(by='endDate', inplace=True, ascending=True)
            group = group.reset_index(drop=True)
            group = group.rename(columns={'endDate': 'date'})
            group['date'] = group['date'].apply(lambda x: x.replace('-', ''))
            group['date'] = group['date'].astype(int)
            # 合并daily信息
            daily_group = daily_data[daily_data['symbol'] == groups[j][0]]
            daily_group = daily_group.reset_index(drop=True)
            group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
            group = group.dropna(axis=0, how='any')
            group = group.reset_index(drop=True)
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
            group.drop(columns='symbol', inplace=True)
            pre_thursday = get_pre_thursday(str(group['date'].iloc[-1]))
            group = group[group['date'] <= int(pre_thursday)]
            # 单独处理下close和marketCap
            max_close = group['close'].max()
            min_close = group['close'].min()
            group['close'] = (group['close'] - min_close) / (max_close - min_close)
            max_marketCap = group['marketCap'].max()
            min_marketCap = group['marketCap'].min()
            group['marketCap'] = (group['marketCap'] - min_marketCap) / (max_marketCap - min_marketCap + 0.00001)
            group = group[-days_input:]
            group = group.reset_index(drop=True)
            # 获得上个的季度末日期
            date_str = datetime.datetime.strptime(str(pre_thursday), '%Y%m%d').date()
            quarter_month = ((date_str.month - 1) // 3) * 3 + 1
            quarter_end_date = datetime.date(date_str.year, quarter_month, 1) + datetime.timedelta(days=-1)
            endDate = int(quarter_end_date.strftime('%Y%m%d'))
            col_names = group.columns.values
            # print(col_names)
            for k in range(1, len(col_names)):
                col_name = col_names[k]
                # 如果不在not_standard_list列表，则进行归一化
                if col_name not in not_standard_list:
                    mean_value = mean_data.loc[mean_data['endDate'] == endDate, col_name].item()
                    std_value = std_data.loc[std_data['endDate'] == endDate, col_name].item()
                    group[col_name] = group[col_name] - mean_value
                    if std_value != 0:
                        group[col_name] = group[col_name] / std_value
                    else:
                        group[col_name] = 0
            group = group.fillna(0)
            group.replace([np.inf, -np.inf], 0, inplace=True)
            #进行预测
            input_data = group.iloc[:, 1:]
            input_data = input_data.values
            input_data = torch.tensor(input_data, device=device)
            input_data = input_data.float()
            input_data = input_data.reshape(1, 768, 125)
            output_data = model(input_data)
            output_data = output_data.cpu()
            output_data = output_data.detach().numpy()
            output_data = pd.DataFrame(output_data[0, :, :])
            group_seg = group.iloc[:, 1:3].join(group.iloc[:, 15:-12])
            output_data.columns = group_seg.columns
            for k in range(1, len(output_data.columns)):
                col_name = output_data.columns[k]
                # 如果不在not_standard_list列表，则进行归一化
                if col_name not in not_standard_list:
                    mean_value = mean_data.loc[mean_data['endDate'] == endDate, col_name].item()
                    std_value = std_data.loc[std_data['endDate'] == endDate, col_name].item()
                    output_data[col_name] = output_data[col_name] * std_value
                    output_data[col_name] = output_data[col_name] + mean_value
                    group_seg[col_name] = group_seg[col_name] * std_value
                    group_seg[col_name] = group_seg[col_name] + mean_value
            output_data.insert(0, 'symbol', [groups[j][0]]*4)
            quarter_end_list = get_next_quarter_end_dates(endDate, 5)
            quarter_end_list = quarter_end_list[1:]
            output_data.insert(0, 'endDate', quarter_end_list)
            today = datetime.datetime.today().strftime("%Y%m%d")
            thursday = get_pre_thursday(today)
            output_data = output_data[output_data['endDate'] < thursday]
            output_data['netAssetValuePerShare'] = (output_data['totalAssets'].astype(float) - output_data[
                'totalDebt'].astype(float)) / (output_data['numberOfShares'].astype(float) + 1.0)
            output_data['revenuePerShare'] = output_data['revenue'].astype(float) / (
                        output_data['numberOfShares'].astype(float) + 1.0)
            output_data['cashFlowPerShare'] = output_data['operatingCashFlow'].astype(float) / (
                        output_data['numberOfShares'].astype(float) + 1.0)
            output_data['debtToEquityRatio'] = output_data['totalDebt'].astype(float) / (
                        output_data['totalStockholdersEquity'].astype(float) + 1.0)
            output_data['quickRatio'] = (output_data['totalCurrentAssets'].astype(float) - output_data[
                'inventory'].astype(float) - output_data['deferredTaxLiabilitiesNonCurrent'].astype(float)) / (
                                                       output_data['totalCurrentLiabilities'].astype(float) + 1.0)
            output_data['earningsMultiple'] = output_data['ebitdaratio']
            output_data['operatingMargin'] = output_data['operatingIncomeRatio']
            output_data['pretaxProfitMargin'] = output_data['incomeBeforeTaxRatio']
            output_data['netProfitMargin'] = output_data['netIncomeRatio']
            output_data['grossProfitMargin'] = output_data['grossProfitRatio']
            output_data['roe'] = output_data['netIncome'].astype(float) / (
                        output_data['totalStockholdersEquity'].astype(float) + 1.0)
            output_data['dcfPerShare'] = 0.0  ##这个地方是重点，我们现在先不给，后面长线系统会专门处理
            column_order = financial_data.columns
            # 重新排序 DataFrame B 的列
            output_data = output_data.reindex(columns=column_order)
            financial_data = pd.concat([financial_data, output_data], axis=0)
            financial_data = financial_data.reset_index(drop=True)
            print(financial_data)
            time.sleep(100)



        financial_data = financial_data.reset_index(drop=True)




if __name__ == '__main__':
    rank_exchange_symbols('Taipei')
    # quarter_end_list = get_next_quarter_end_dates('20230630')
    # print(quarter_end_list)
    # today = datetime.datetime.today().strftime("%Y%m%d")
    # thursday = get_pre_thursday(today)
    # print(thursday)
