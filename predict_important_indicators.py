import tushare as ts
import tushare_tools as tt
import random
import os
import tools
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

input_data_name_list = ['fina_indicator', 'income', 'balancesheet', 'cashflow', 'daily', 'daily_basic', 'moneyflow', 'stk_factor', 'weekly', 'monthly']
input_data_drop_name_list = [['ts_code', 'ann_date', 'end_date'],
                             ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'],
                             ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'],
                             ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'],
                             ['ts_code', 'trade_date'],
                             ['ts_code', 'trade_date'],
                             ['ts_code', 'trade_date'],
                             ['ts_code', 'trade_date'],
                             ['ts_code', 'trade_date'],
                             ['ts_code', 'trade_date'],]

def generate_scaler(folder):
    folder_paths = tools.get_folders_path(folder)
    for i in range(len(folder_paths)):
        folder_path = folder_paths[i]
        csv_files = tools.get_specified_files(folder_path, '.csv')
        data_list = []
        for j in range(len(csv_files)):
            data_frame = pd.read_csv(csv_files[j])
            data_list.append(data_frame)
        data = pd.concat(data_list, axis=0)
        data = data.drop('Unnamed: 0', axis='columns')

        data = data.drop('ts_code', axis='columns')
        data = data.reset_index(drop=True)
        data = data.fillna(0.0)
        scaler = StandardScaler()
        print(data)
        data = scaler.fit_transform(data)
        dump(scaler, folder_path + '/std_scaler.bin', compress=True)
        print(data.shape)
        input()

def download_input_data(ts_code):
    folder = 'input_data'
    if (os.path.exists(folder)) == False:
        os.mkdir(folder)
    five_mininte_data_folder = 'input_data/five_mininte'
    if (os.path.exists(five_mininte_data_folder)) == False:
        os.mkdir(five_mininte_data_folder)
    thirty_mininte_data_folder = 'input_data/thirty_mininte'
    if (os.path.exists(thirty_mininte_data_folder)) == False:
        os.mkdir(thirty_mininte_data_folder)
    daily_data_folder = 'input_data/daily'
    if (os.path.exists(daily_data_folder)) == False:
        os.mkdir(daily_data_folder)
    daily_basic_data_folder = 'input_data/daily_basic'
    if (os.path.exists(daily_basic_data_folder)) == False:
        os.mkdir(daily_basic_data_folder)
    moneyflow_data_folder = 'input_data/moneyflow'
    if (os.path.exists(moneyflow_data_folder)) == False:
        os.mkdir(moneyflow_data_folder)
    stk_factor_data_folder = 'input_data/stk_factor'
    if (os.path.exists(stk_factor_data_folder)) == False:
        os.mkdir(stk_factor_data_folder)
    weekly_data_folder = 'input_data/weekly'
    if (os.path.exists(weekly_data_folder)) == False:
        os.mkdir(weekly_data_folder)
    monthly_data_folder = 'input_data/monthly'
    if (os.path.exists(monthly_data_folder)) == False:
        os.mkdir(monthly_data_folder)
    income_data_folder = 'input_data/income'
    if (os.path.exists(income_data_folder)) == False:
        os.mkdir(income_data_folder)
    balancesheet_data_folder = 'input_data/balancesheet'
    if (os.path.exists(balancesheet_data_folder)) == False:
        os.mkdir(balancesheet_data_folder)
    cashflow_data_folder = 'input_data/cashflow'
    if (os.path.exists(cashflow_data_folder)) == False:
        os.mkdir(cashflow_data_folder)
    fina_indicator_data_folder = 'input_data/fina_indicator'
    if (os.path.exists(fina_indicator_data_folder)) == False:
        os.mkdir(fina_indicator_data_folder)

    # five_mininte_data = tt.get_5_min_data(ts_code, trade_date)
    # five_mininte_data = five_mininte_data.drop(['ts_code', 'trade_time', 'trade_date'], axis=1)
    # five_mininte_data.to_csv(five_mininte_data_folder + '/' + ts_code + '_' + trade_date + '.csv', mode = 'w')
    #
    # thirty_mininte_data = tt.get_30_min_data(ts_code, trade_date)
    # thirty_mininte_data = thirty_mininte_data.drop(['ts_code', 'trade_time', 'trade_date'], axis=1)
    # thirty_mininte_data.to_csv(thirty_mininte_data_folder + '/' + ts_code + '_' + trade_date + '.csv', mode = 'w')

    # daily_data = pro.daily(ts_code=ts_code)
    # daily_data.to_csv(daily_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # daily_basic_data = pro.daily_basic(ts_code=ts_code)
    # daily_basic_data = daily_basic_data.fillna(0.0)
    # daily_basic_data.to_csv(daily_basic_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # moneyflow_data = pro.moneyflow(ts_code=ts_code)
    # moneyflow_data = moneyflow_data.fillna(0.0)
    # moneyflow_data.to_csv(moneyflow_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # stk_factor_data = pro.stk_factor(ts_code=ts_code)
    # stk_factor_data = stk_factor_data.fillna(0.0)
    # stk_factor_data.to_csv(stk_factor_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # hs300_data = tt.get_index_daily_data('399300.SZ', trade_date)
    # hs300_data = hs300_data.drop(['ts_code', 'trade_date'], axis=1)
    # hs300_data.to_csv(hs300_data_folder + '/' + ts_code + '_' + trade_date + '.csv', mode = 'w')
    #
    # weekly_data = pro.weekly(ts_code=ts_code)
    # weekly_data.to_csv(weekly_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # monthly_data = pro.monthly(ts_code=ts_code)
    # monthly_data.to_csv(monthly_data_folder + '/' + ts_code + '.csv', mode = 'w')
    #
    # income_data =  pro.income(ts_code=ts_code)
    # income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
    # income_data = income_data.reset_index(drop=True)
    # income_data = income_data.fillna(0.0)
    # income_data.to_csv(income_data_folder + '/' + ts_code + '.csv', mode = 'w')
    # #
    # balancesheet_data =  pro.balancesheet(ts_code=ts_code)
    # balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
    # balancesheet_data = balancesheet_data.reset_index(drop=True)
    # balancesheet_data = balancesheet_data.fillna(0.0)
    # balancesheet_data.to_csv(balancesheet_data_folder + '/' + ts_code + '.csv', mode = 'w')
    # #
    # cashflow_data =  pro.cashflow(ts_code=ts_code)
    # cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
    # cashflow_data = cashflow_data.reset_index(drop=True)
    # cashflow_data = cashflow_data.fillna(0.0)
    # cashflow_data.to_csv(cashflow_data_folder + '/' + ts_code + '.csv', mode = 'w')
    # #
    # fina_indicator_data =  pro.fina_indicator(ts_code=ts_code)
    # fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
    # fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    # fina_indicator_data = fina_indicator_data.fillna(0.0)
    # fina_indicator_data.to_csv(fina_indicator_data_folder + '/' + ts_code + '.csv', mode = 'w')

def get_input_data(ts_code, start_date, end_date):
    trade_date_list = tt.get_trade_date_list_period(start_date, end_date)
    for i in range(len(trade_date_list)):
        trade_date = trade_date_list[i]
        generate_input_data(ts_code, trade_date)
        print(float(i)/len(trade_date_list))
        print(trade_date)
        print(ts_code)

##
def get_forcast_label(ts_code, trade_date):
    try:
        start_date = tools.get_delta_date(trade_date, -365)
        end_date = tools.get_delta_date(trade_date, 365)
        fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
        fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
        fina_indicator_data = fina_indicator_data.reset_index(drop=True)
        data_future = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['end_date'] < trade_date].index)
        data_future = data_future.reset_index(drop=True)
        tail_num = data_future.shape[0]
        q_sales_yoy_future = data_future['q_sales_yoy'][tail_num-1]
        q_dt_roe_future = data_future['q_dt_roe'][tail_num-1]
        grossprofit_margin_future = data_future['grossprofit_margin'][tail_num-1]
        ocf_yoy_future = data_future['ocf_yoy'][tail_num-1]

        data_past = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['end_date'] >= trade_date].index)
        data_past = data_past.reset_index(drop=True)
        q_sales_yoy_past = data_past['q_sales_yoy'][0]
        q_dt_roe_past = data_past['q_dt_roe'][0]
        grossprofit_margin_past = data_past['grossprofit_margin'][0]
        ocf_yoy_past = data_past['ocf_yoy'][0]

        if (q_sales_yoy_future > q_sales_yoy_past) and (q_dt_roe_future > q_dt_roe_past) and (grossprofit_margin_future > grossprofit_margin_past) and (ocf_yoy_future > ocf_yoy_past):
            return 2
        elif (ocf_yoy_future > ocf_yoy_past) and (q_sales_yoy_future > q_sales_yoy_past):
            return 1
        else:
            return 0
    except:
        return -1

def get_middle_system_data():
    folder = 'middle_system'
    if (os.path.exists(folder)) == False:
        os.mkdir(folder)
    folder_train = 'middle_system/train'
    if (os.path.exists(folder_train)) == False:
        os.mkdir(folder_train)
    folder_train = 'middle_system/train/0'
    if (os.path.exists(folder_train)) == False:
        os.mkdir(folder_train)
    folder_train = 'middle_system/train/1'
    if (os.path.exists(folder_train)) == False:
        os.mkdir(folder_train)
    folder_train = 'middle_system/train/2'
    if (os.path.exists(folder_train)) == False:
        os.mkdir(folder_train)
    today = tools.get_today()
    end_date = tools.get_delta_date(today, -183)
    # #加载daily_basic数据，选择股息率ttm>0的数据保留
    #
    # folder_path = 'input_data/daily_basic'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['dv_ttm'] < 0.0001].index)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.reset_index(drop=True)
    # print(data)
    # # data.to_csv(folder + '/daily_basic.csv', mode='w')
    # # daily_basic_data = pd.read_csv(folder + '/daily_basic.csv')
    # # daily_basic_data = daily_basic_data.drop('Unnamed: 0', axis='columns')
    # # print(daily_basic_data)
    # #加载daily数据
    # folder_path = 'input_data/daily'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # daily_data = pd.concat(data_list, axis=0)
    # daily_data = daily_data.drop('Unnamed: 0', axis='columns')
    # daily_data = daily_data.reset_index(drop=True)
    # data=pd.merge(daily_data,data,on=['ts_code','trade_date'])
    # print(data)
    #
    # # daily_data = pd.read_csv(folder + '/daily_data.csv')
    # # daily_data = daily_data.drop('Unnamed: 0', axis='columns')
    # # print(daily_data)
    # #加载moneyflow数据
    # folder_path = 'input_data/moneyflow'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # moneyflow_data = pd.concat(data_list, axis=0)
    # moneyflow_data = moneyflow_data.drop('Unnamed: 0', axis='columns')
    # moneyflow_data = moneyflow_data.reset_index(drop=True)
    # data=pd.merge(moneyflow_data,data, on=['ts_code','trade_date'])
    # print(data)
    #
    # #加载stk_factor数据
    # folder_path = 'input_data/stk_factor'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # stk_factor_data = pd.concat(data_list, axis=0)
    # stk_factor_data = stk_factor_data.drop('Unnamed: 0', axis='columns')
    # stk_factor_data = stk_factor_data.reset_index(drop=True)
    # data=pd.merge(stk_factor_data,data,on=['ts_code','trade_date'])
    # print(data)
    # data.to_csv(folder + '/daily_data.csv', mode='w')
    #
    # ##加载weekly数据
    # folder_path = 'input_data/weekly'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.reset_index(drop=True)
    # print(data)
    # data.to_csv(folder + '/weekly_data.csv', mode='w')
    #
    # ##加载monthly数据
    # folder_path = 'input_data/monthly'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_frame = data_frame.drop(data_frame[data_frame['trade_date'] > int(end_date)].index)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.reset_index(drop=True)
    # print(data)
    # data.to_csv(folder + '/monthly_data.csv', mode='w')
    #
    # ##加载财务数据
    # folder_path = 'input_data/income'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.drop(['ann_date', 'f_ann_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
    # income_data = data.reset_index(drop=True)
    # print(income_data)
    #
    # folder_path = 'input_data/balancesheet'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.drop(['ann_date', 'f_ann_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
    # balancesheet_data = data.reset_index(drop=True)
    # print(balancesheet_data)
    # financial_data = pd.merge(income_data,balancesheet_data, on=['ts_code','end_date'])
    # print(financial_data)
    #
    # folder_path = 'input_data/cashflow'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.drop(['ann_date', 'f_ann_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
    # cashflow_data = data.reset_index(drop=True)
    # print(cashflow_data)
    # financial_data = pd.merge(financial_data,cashflow_data, on=['ts_code','end_date'])
    # print(financial_data)
    #
    # folder_path = 'input_data/fina_indicator'
    # csv_files = tools.get_specified_files(folder_path, '.csv')
    # data_list = []
    # for i in range(len(csv_files)):
    #     print(float(i)/len(csv_files))
    #     data_frame = pd.read_csv(csv_files[i])
    #     if data_frame.empty:
    #         pass
    #     else:
    #         data_frame = data_frame.fillna(0.0)
    #         data_list.append(data_frame)
    # data = pd.concat(data_list, axis=0)
    # data = data.drop('Unnamed: 0', axis='columns')
    # data = data.drop(['ann_date'], axis='columns')
    # fina_indicator_data = data.reset_index(drop=True)
    # print(fina_indicator_data)
    # financial_data = pd.merge(financial_data,fina_indicator_data, on=['ts_code','end_date'])
    # print(financial_data)
    # financial_data.to_csv(folder + '/financial_data.csv', mode='w')
    #########随机生成训练数据
    daily_data = pd.read_csv(folder + '/daily_data.csv')
    # daily_data = daily_data.drop('Unnamed: 0', axis='columns')
    # weekly_data = pd.read_csv(folder + '/weekly_data.csv')
    # # weekly_data = weekly_data.drop('Unnamed: 0', axis='columns')
    # monthly_data = pd.read_csv(folder + '/monthly_data.csv')
    # # monthly_data = weekly_data.drop('Unnamed: 0', axis='columns')
    # financial_data = pd.read_csv(folder + '/financial_data.csv')
    # # financial_data = weekly_data.drop('Unnamed: 0', axis='columns')
    count = 0
    num = 100000
    while count < num:
        try:
            ##生成日线数据
            i = random.randint(0, daily_data.shape[0]-64)
            data_frame = daily_data.iloc[i:i+64]
            data_frame = data_frame.drop(['Unnamed: 0'], axis='columns')
            data_frame = data_frame.reset_index(drop=True)
            if data_frame['ts_code'][0] != data_frame['ts_code'][63]:
                pass
            else:
                ts_code = data_frame['ts_code'][0]
                trade_date = str(data_frame['trade_date'][0])

                # #添加weekly数据
                start_date = tools.get_delta_date(trade_date, -640)
                end_date = trade_date
                weekly_data = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
                weekly_data = weekly_data.head(64)
                weekly_data = weekly_data.drop(['ts_code','trade_date'], axis='columns')
                data_frame = pd.concat([data_frame, weekly_data], axis=1)
                data_frame = data_frame.fillna(0.0)
                # #添加monthly数据
                start_date = tools.get_delta_date(trade_date, -3000)
                end_date = trade_date
                monthly_data = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=end_date)
                monthly_data = monthly_data.head(64)
                monthly_data = monthly_data.drop(['ts_code','trade_date'], axis='columns')
                data_frame = pd.concat([data_frame, monthly_data], axis=1)
                data_frame = data_frame.fillna(0.0)
                data_frame = data_frame.drop(['ts_code','trade_date'], axis='columns')
                data_frame.columns = range(len(data_frame.columns))
                # print(data_frame)
                ##添加income数据
                start_date = tools.get_delta_date(trade_date, -2000)
                end_date = trade_date
                income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
                income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
                income_data = income_data.reset_index(drop=True)
                income_data = income_data.head(16)
                income_data = income_data.drop(['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
                income_data.columns = range(len(income_data.columns))
                data_frame = pd.concat([data_frame, income_data], axis=0)
                data_frame = data_frame.fillna(0.0)
                ##添加balancesheet数据
                start_date = tools.get_delta_date(trade_date, -2000)
                end_date = trade_date
                balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
                balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
                balancesheet_data = balancesheet_data.reset_index(drop=True)
                balancesheet_data = balancesheet_data.head(16)
                balancesheet_data = balancesheet_data.drop(['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
                balancesheet_data.columns = range(len(balancesheet_data.columns))
                data_frame = pd.concat([data_frame, balancesheet_data], axis=0)
                data_frame = data_frame.fillna(0.0)
                ##添加cashflow数据
                start_date = tools.get_delta_date(trade_date, -2000)
                end_date = trade_date
                cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
                cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
                cashflow_data = cashflow_data.reset_index(drop=True)
                cashflow_data = cashflow_data.head(16)
                cashflow_data = cashflow_data.drop(['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'end_type', 'update_flag'], axis='columns')
                cashflow_data.columns = range(len(cashflow_data.columns))
                data_frame = pd.concat([data_frame, cashflow_data], axis=0)
                data_frame = data_frame.fillna(0.0)
                ##添加fina_indicator数据
                start_date = tools.get_delta_date(trade_date, -2000)
                end_date = trade_date
                fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
                fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
                fina_indicator_data = fina_indicator_data.reset_index(drop=True)
                fina_indicator_data = fina_indicator_data.head(16)
                fina_indicator_data = fina_indicator_data.drop(['ts_code', 'ann_date', 'end_date'], axis='columns')
                fina_indicator_data.columns = range(len(fina_indicator_data.columns))
                data_frame = pd.concat([data_frame, fina_indicator_data], axis=0)
                data_frame = data_frame.fillna(0.0)
                data_frame = data_frame.reset_index(drop=True)
                # data_frame
                if data_frame.shape[0] == 128:
                    label = get_forcast_label(ts_code, trade_date)
                    if label == 0:
                        data_frame_name = folder + '/train/0/' + ts_code + '_' + str(trade_date) + '.csv'
                        data_frame.to_csv(data_frame_name, mode='w')
                    elif label == 1:
                        data_frame_name = folder + '/train/1/' + ts_code + '_' + str(trade_date) + '.csv'
                        data_frame.to_csv(data_frame_name, mode='w')
                    elif label == 2:
                        data_frame_name = folder + '/train/2/' + ts_code + '_' + str(trade_date) + '.csv'
                        data_frame.to_csv(data_frame_name, mode='w')
                    count = count + 1
                    print(count)
        except:
            pass

if __name__ == '__main__':
    # start_date = '20100101'
    # end_date = '2020101'
    # ts_code_list = tt.get_ts_code_list_all()
    # for i in range(len(ts_code_list)):
    #     ts_code = ts_code_list[i]
    #     generate_input_data(ts_code)
    #     print('***************************')
    #     print(float(i)/len(ts_code_list))
    # df = pro.income(ts_code='601919.SH')
    # print(df)
    # generate_scaler('input_data')
    get_middle_system_data()
    # print(get_forcast_label('601919.SH', '20200101'))

