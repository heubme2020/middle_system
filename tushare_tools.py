import tools
import time
import datetime
import pandas as pd
import tushare as ts
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

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
def get_trade_date_list(trade_date, trade_date_num):
    trade_date_list = []
    today_date = tools.get_today()
    if trade_date > today_date:
        trade_date = today_date
    start_date = tools.get_delta_date(trade_date, -3 * trade_date_num)
    trade_cal = pro.trade_cal(start_date=start_date, end_date=trade_date)
    trade_cal = trade_cal.drop(trade_cal[trade_cal['is_open'] == 0].index)
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    for i in range(trade_date_num):
        trade_date_list.append(trade_cal['cal_date'].iloc[i])
    return trade_date_list

def get_trade_date_list_period(start_date, end_date):
    trade_date_list = []
    trade_cal = pro.trade_cal(start_date=start_date, end_date=end_date)
    trade_cal = trade_cal.drop(trade_cal[trade_cal['is_open'] == 0].index)
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    for i in range(trade_cal.shape[0]):
        trade_date_list.append(trade_cal['cal_date'].iloc[i])
    return trade_date_list

##往後推3天
def get_season_date_list_period(start_date, end_date):
    season_date_list = []
    trade_date_list = get_trade_date_list_period(start_date, end_date)
    for i in range(len(trade_date_list)):
        trade_date = trade_date_list[i]
        season_start, season_end = tools.get_season_border(trade_date, -1)
        season_date = tools.get_delta_date(season_end, 3)
        if season_date not in season_date_list:
            season_date_list.append(season_date)
    return season_date_list

def get_ts_code_list_all():
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = pd.concat([ts_code_data_L, ts_code_data_D, ts_code_data_P])
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    ts_code_data = ts_code_data.reset_index(drop=True)
    return list(ts_code_data['ts_code'])


def get_ts_code_list(trade_date):
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = pd.concat([ts_code_data_L, ts_code_data_D,ts_code_data_P])
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    ts_code_data = ts_code_data.reset_index(drop=True)
    ts_code_data = ts_code_data.drop(ts_code_data[ts_code_data['list_date'] > trade_date].index)
    return ts_code_data['ts_code']

def get_season_list():
    season_list = []
    pre_year = int(tools.get_toyear() - 1)
    for i in range(1995, pre_year):
        season_list.append(str(i) + '0101')
        season_list.append(str(i) + '0331')
        season_list.append(str(i) + '0401')
        season_list.append(str(i) + '0630')
        season_list.append(str(i) + '0701')
        season_list.append(str(i) + '0930')
        season_list.append(str(i) + '1001')
        season_list.append(str(i) + '1231')
    return season_list



def get_fina_indicator_data_period(ts_code, start_date, end_date, season_num = 3):
    fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
    fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    fina_indicator_data = fina_indicator_data.fillna(0.0)
    if fina_indicator_data.shape[0] >= season_num:
        fina_indicator_data = fina_indicator_data.head(season_num)
    return fina_indicator_data
#
# def get_fina_indicator_data_delta(ts_code, trade_date):
#     start_date = tools.get_delta_date(trade_date, -200)
#     end_date = tools.get_delta_date(trade_date, 200)
#     fina_indicator_data_pre = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=trade_date)
#     fina_indicator_data_pre = fina_indicator_data_pre.drop_duplicates(subset=['end_date'], keep='first')
#     fina_indicator_data_pre = fina_indicator_data_pre.reset_index(drop=True)
#     fina_indicator_data_pre = fina_indicator_data_pre.fillna(0.0)
#     fina_indicator_data_next = pro.fina_indicator(ts_code=ts_code, start_date=trade_date, end_date=end_date)
#     fina_indicator_data_next = fina_indicator_data_next.drop_duplicates(subset=['end_date'], keep='first')
#     fina_indicator_data_next = fina_indicator_data_next.reset_index(drop=True)
#     fina_indicator_data_next = fina_indicator_data_next.fillna(0.0)


def get_current_season_fina_indicator_data(season_num=3):
    date_today = tools.get_today()
    hour = time.strftime('%H', time.localtime())
    if int(hour) > 17:
        date_today = tools.get_delta_date(date_today, -1)
    start_date = tools.get_delta_date(date_today, -365)
    end_date = date_today
    ts_code_data = pro.stock_basic(exchange='', list_status='L')
    ts_codes = ts_code_data['ts_code']
    season_fina_indicator_data = []
    for i in range(ts_code_data.shape[0]):
        print('读取财报数据：' + str(float(i)/ts_code_data.shape[0]))
        ts_code = ts_codes[i]
        fina_indicator_data = get_fina_indicator_data_period(ts_code, start_date, end_date, season_num)
        season_fina_indicator_data.append(fina_indicator_data)
        time.sleep(0.01)
    # data = pd.concat([data0, data1, data2])
    current_season_fina_indicator_data = pd.concat(season_fina_indicator_data, axis=0)
    current_season_fina_indicator_data = current_season_fina_indicator_data.reset_index(drop=True)
    current_season_fina_indicator_data.to_csv('current_season_fina_indicator_data.csv')

##行情数据相关
def get_5_min_data(ts_code, trade_date, n=4):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    start_date = start_date + ' 09:00:00'
    start_hour = str(datetime.datetime.strptime(start_date, '%Y%m%d %H:%M:%S'))
    end_date = end_date + ' 17:00:00'
    end_hour = str(datetime.datetime.strptime(end_date, '%Y%m%d %H:%M:%S'))
    five_mininte_data = ts.pro_bar(ts_code=ts_code, freq='5min', start_date=start_hour, end_date=end_hour)
    five_mininte_data = five_mininte_data.head(int(n * 49))
    five_mininte_data = five_mininte_data.fillna(0.0)
    return five_mininte_data

def get_30_min_data(ts_code, trade_date, n=16):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    start_date = start_date + ' 09:00:00'
    start_hour = str(datetime.datetime.strptime(start_date, '%Y%m%d %H:%M:%S'))
    end_date = end_date + ' 17:00:00'
    end_hour = str(datetime.datetime.strptime(end_date, '%Y%m%d %H:%M:%S'))
    half_hour_data = ts.pro_bar(ts_code=ts_code, freq='30min', start_date=start_hour, end_date=end_hour)
    half_hour_data = half_hour_data.head(int(n * 9))
    half_hour_data = half_hour_data.fillna(0.0)
    return half_hour_data


def get_daily_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    daily_data = daily_data.head(n)
    daily_data = daily_data.fillna(0.0)
    return daily_data


def get_daily_basic_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    daily_basic_data = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
    daily_basic_data = daily_basic_data.head(n)
    daily_basic_data = daily_basic_data.fillna(0.0)
    return daily_basic_data


def get_moneyflow_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    moneyflow_data = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    moneyflow_data = moneyflow_data.head(n)
    moneyflow_data = moneyflow_data.fillna(0.0)
    return moneyflow_data

def get_index_daily_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -2*(n+3))
    end_date = trade_date
    index_daily_data = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    index_daily_data = index_daily_data.head(n)
    index_daily_data = index_daily_data.fillna(0.0)
    return index_daily_data

def get_weekly_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -10*n)
    end_date = trade_date
    weekly_data = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
    weekly_data = weekly_data.head(n)
    weekly_data = weekly_data.fillna(0.0)
    return weekly_data


def get_monthly_data(ts_code, trade_date, n=64):
    start_date = tools.get_delta_date(trade_date, -40*n)
    end_date = trade_date
    monthly_data = pro.monthly(ts_code=ts_code, start_date=start_date, end_date=end_date)
    monthly_data = monthly_data.head(n)
    monthly_data = monthly_data.fillna(0.0)
    return monthly_data

# 获取财务数据
def get_income_data(ts_code, trade_date, n=16):
    start_date = tools.get_delta_date(trade_date, -100*n)
    end_date = trade_date
    income_data = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
    income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
    income_data = income_data.reset_index(drop=True)
    income_data = income_data.head(n)
    income_data = income_data.fillna(0.0)
    return income_data

def get_balancesheet_data(ts_code, trade_date, n=16):
    start_date = tools.get_delta_date(trade_date, -100*n)
    end_date = trade_date
    balancesheet_data = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
    balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
    balancesheet_data = balancesheet_data.reset_index(drop=True)
    balancesheet_data = balancesheet_data.head(n)
    balancesheet_data = balancesheet_data.fillna(0.0)
    return balancesheet_data


def get_cashflow_data(ts_code, trade_date, n=16):
    start_date = tools.get_delta_date(trade_date, -100*n)
    end_date = trade_date
    cashflow_data = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
    cashflow_data = cashflow_data.reset_index(drop=True)
    cashflow_data = cashflow_data.head(n)
    cashflow_data = cashflow_data.fillna(0.0)
    return cashflow_data

def get_fina_indicator_data(ts_code, trade_date, n=16):
    # trade_date_list = get_trade_date_list(trade_date, n * 100)
    start_date = tools.get_delta_date(trade_date, -100*n)
    end_date = trade_date
    fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date,end_date=end_date)
    fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    fina_indicator_data = fina_indicator_data.head(n)
    fina_indicator_data = fina_indicator_data.fillna(0.0)
    return fina_indicator_data

def get_dv_ttm(ts_code, trade_date):
    start_date = tools.get_delta_date(trade_date, -30)
    end_date = trade_date
    daily_basic_data = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
    daily_basic_data = daily_basic_data.fillna(0.0)
    if daily_basic_data.empty:
        return 0.0
    else:
        dv_ttm = daily_basic_data['dv_ttm'][0]
        return dv_ttm

if __name__ == "__main__":
    # Let's build our model
    # train(num_epochs=2000, batch_size=32)
    # fina_indicator_data = pro.fina_indicator(ts_code = '601919.SH')
    # season_list = get_season_list()
    # print(season_list)
    # get_current_season_fina_indicator_data()
    df = pro.forecast_vip(period='20230331')
    df.to_csv('forecast.csv', mode='w')
    print(df)

