from urllib.request import urlopen
import ssl
import os
import time
import certifi
import json
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
token ='6263d89930d3ba4b1329d603814270ad'

def get_jsonparsed_data(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    # response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)
def get_all_stock_list():
    url = "https://financialmodelingprep.com/api/v3/stock/list?apikey="+token
    stock_all = get_jsonparsed_data(url)
    return stock_all
def get_specific_country_stock_list(country):
    url = "https://financialmodelingprep.com/api/v3/stock-screener?limit=100000&country=" + country + "&apikey="+token
    stock_list = get_jsonparsed_data(url)
    return stock_list
def get_specific_exchange_stock_list(suffix):
    url = "https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey="+token
    stock_all = get_jsonparsed_data(url)
    # print(stock_all)
    stock_list = []
    for i in range(len(stock_all)):
        stock = stock_all[i]
        # print(stock)
        if stock.endswith(suffix):
            stock_list.append(stock)
    return stock_list

def get_income_quarter_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock +'?period=quarter&limit=400&apikey='+token
    income_quarter_sheets = get_jsonparsed_data(url)
    return income_quarter_sheets
def get_income_annual_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/income-statement/'+stock +'?limit=120&apikey='+token
    income_quarter_sheets = get_jsonparsed_data(url)
    return income_quarter_sheets
def get_balance_sheet_quarter_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+stock +'?period=quarter&limit=400&apikey='+token
    balance_sheet_quarter_sheets = get_jsonparsed_data(url)
    return balance_sheet_quarter_sheets
def get_balance_sheet_annual_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/balance-sheet-statement/'+stock +'?limit=120&apikey='+token
    balance_sheet_quarter_sheets = get_jsonparsed_data(url)
    return balance_sheet_quarter_sheets
def get_cash_flow_quarter_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/cash-flow-statement/'+stock +'?period=quarter&limit=400&apikey='+token
    cash_flow_quarter_sheets = get_jsonparsed_data(url)
    return cash_flow_quarter_sheets
def get_cash_flow_annual_sheets(stock):
    url = 'https://financialmodelingprep.com/api/v3/cash-flow-statement/'+stock +'?limit=120&apikey='+token
    cash_flow_quarter_sheets = get_jsonparsed_data(url)
    return cash_flow_quarter_sheets
def get_financial_sheets(exchange_suffix, folder_name):
    stock_list = get_specific_exchange_stock_list(exchange_suffix)
    stock_num = len(stock_list)
    print(stock_num)
    if (os.path.exists(folder_name)) == False:
        os.mkdir(folder_name)
    annual_name = folder_name + '/annual'
    if (os.path.exists(annual_name)) == False:
        os.mkdir(annual_name)
    quarter_name = folder_name + '/quarter'
    if (os.path.exists(quarter_name)) == False:
        os.mkdir(quarter_name)
    for i in range(stock_num):
        print(float(i)/stock_num)
        stock = stock_list[i]
        # time.sleep(0.1)
        #下载income数据
        income_quarter_name = quarter_name + '/' + stock + '--income_quarter_data.csv'
        if (os.path.exists(income_quarter_name)) == False:
            try:
                income_quarter_sheets = get_income_quarter_sheets(stock)
                income_quarter_sheet = income_quarter_sheets[0]
                income_quarter_sheet['date'] = income_quarter_sheet['date'].replace('-', '')
                income_quarter_sheet['fillingDate'] = income_quarter_sheet['fillingDate'].replace('-', '')
                income_quarter_sheet['acceptedDate'] = income_quarter_sheet['acceptedDate'].replace('-', '')
                income_quarter_dataframe = pd.DataFrame([income_quarter_sheet])
                income_quarter_dataframe.to_csv(income_quarter_name, index=False, mode='w')
                for j in range(1, len(income_quarter_sheets)):
                    income_quarter_sheet = income_quarter_sheets[j]
                    income_quarter_sheet['date'] = income_quarter_sheet['date'].replace('-', '')
                    income_quarter_sheet['fillingDate'] = income_quarter_sheet['fillingDate'].replace('-', '')
                    income_quarter_sheet['acceptedDate'] = income_quarter_sheet['acceptedDate'].replace('-', '')
                    income_quarter_dataframe = pd.DataFrame([income_quarter_sheet])
                    income_quarter_dataframe.to_csv(income_quarter_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')
        income_annual_name = annual_name + '/' + stock + '--income_annual_data.csv'
        if (os.path.exists(income_annual_name)) == False:
            try:
                income_annual_sheets = get_income_annual_sheets(stock)
                income_annual_sheet = income_annual_sheets[0]
                income_annual_sheet['date'] = income_annual_sheet['date'].replace('-', '')
                income_annual_sheet['fillingDate'] = income_annual_sheet['fillingDate'].replace('-', '')
                income_annual_sheet['acceptedDate'] = income_annual_sheet['acceptedDate'].replace('-', '')
                income_annual_dataframe = pd.DataFrame([income_annual_sheet])
                income_annual_dataframe.to_csv(income_annual_name, index=False, mode='w')
                for j in range(1, len(income_annual_sheets)):
                    income_annual_sheet = income_annual_sheets[j]
                    income_annual_sheet['date'] = income_annual_sheet['date'].replace('-', '')
                    income_annual_sheet['fillingDate'] = income_annual_sheet['fillingDate'].replace('-', '')
                    income_annual_sheet['acceptedDate'] = income_annual_sheet['acceptedDate'].replace('-', '')
                    income_annual_dataframe = pd.DataFrame([income_annual_sheet])
                    income_annual_dataframe.to_csv(income_annual_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')
        #下载balance-sheet数据
        balance_sheet_quarter_name = quarter_name + '/' + stock + '--balance_sheet_quarter_data.csv'
        if (os.path.exists(balance_sheet_quarter_name)) == False:
            try:
                balance_sheet_quarter_sheets = get_balance_sheet_quarter_sheets(stock)
                balance_sheet_quarter_sheet = balance_sheet_quarter_sheets[0]
                balance_sheet_quarter_sheet['date'] = balance_sheet_quarter_sheet['date'].replace('-', '')
                balance_sheet_quarter_sheet['fillingDate'] = balance_sheet_quarter_sheet['fillingDate'].replace('-', '')
                balance_sheet_quarter_sheet['acceptedDate'] = balance_sheet_quarter_sheet['acceptedDate'].replace('-', '')
                balance_sheet_quarter_dataframe = pd.DataFrame([balance_sheet_quarter_sheet])
                balance_sheet_quarter_dataframe.to_csv(balance_sheet_quarter_name, index=False, mode='w')
                for j in range(1, len(balance_sheet_quarter_sheets)):
                    balance_sheet_quarter_sheet = balance_sheet_quarter_sheets[j]
                    balance_sheet_quarter_sheet['date'] = balance_sheet_quarter_sheet['date'].replace('-', '')
                    balance_sheet_quarter_sheet['fillingDate'] = balance_sheet_quarter_sheet['fillingDate'].replace('-', '')
                    balance_sheet_quarter_sheet['acceptedDate'] = balance_sheet_quarter_sheet['acceptedDate'].replace('-', '')
                    balance_sheet_quarter_dataframe = pd.DataFrame([balance_sheet_quarter_sheet])
                    balance_sheet_quarter_dataframe.to_csv(balance_sheet_quarter_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')
        balance_sheet_annual_name = annual_name + '/' + stock + '--balance_sheet_annual_data.csv'
        if (os.path.exists(balance_sheet_annual_name)) == False:
            try:
                balance_sheet_annual_sheets = get_balance_sheet_annual_sheets(stock)
                balance_sheet_annual_sheet = balance_sheet_annual_sheets[0]
                balance_sheet_annual_sheet['date'] = balance_sheet_annual_sheet['date'].replace('-', '')
                balance_sheet_annual_sheet['fillingDate'] = balance_sheet_annual_sheet['fillingDate'].replace('-', '')
                balance_sheet_annual_sheet['acceptedDate'] = balance_sheet_annual_sheet['acceptedDate'].replace('-', '')
                balance_sheet_annual_dataframe = pd.DataFrame([balance_sheet_annual_sheet])
                balance_sheet_annual_dataframe.to_csv(balance_sheet_annual_name, index=False, mode='w')
                for j in range(1, len(balance_sheet_annual_sheets)):
                    balance_sheet_annual_sheet = balance_sheet_annual_sheets[j]
                    balance_sheet_annual_sheet['date'] = balance_sheet_annual_sheet['date'].replace('-', '')
                    balance_sheet_annual_sheet['fillingDate'] = balance_sheet_annual_sheet['fillingDate'].replace('-', '')
                    balance_sheet_annual_sheet['acceptedDate'] = balance_sheet_annual_sheet['acceptedDate'].replace('-', '')
                    balance_sheet_annual_dataframe = pd.DataFrame([balance_sheet_annual_sheet])
                    balance_sheet_annual_dataframe.to_csv(balance_sheet_annual_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')
        #下载cash-flow数据
        cash_flow_quarter_name = quarter_name + '/' + stock + '--cash_flow_quarter_data.csv'
        if (os.path.exists(cash_flow_quarter_name)) == False:
            try:
                cash_flow_quarter_sheets = get_cash_flow_quarter_sheets(stock)
                cash_flow_quarter_sheet = cash_flow_quarter_sheets[0]
                cash_flow_quarter_sheet['date'] = cash_flow_quarter_sheet['date'].replace('-', '')
                cash_flow_quarter_sheet['fillingDate'] = cash_flow_quarter_sheet['fillingDate'].replace('-', '')
                cash_flow_quarter_sheet['acceptedDate'] = cash_flow_quarter_sheet['acceptedDate'].replace('-', '')
                cash_flow_quarter_dataframe = pd.DataFrame([cash_flow_quarter_sheet])
                cash_flow_quarter_dataframe.to_csv(cash_flow_quarter_name, index=False, mode='w')
                for j in range(1, len(cash_flow_quarter_sheets)):
                    cash_flow_quarter_sheet = cash_flow_quarter_sheets[j]
                    cash_flow_quarter_sheet['date'] = cash_flow_quarter_sheet['date'].replace('-', '')
                    cash_flow_quarter_sheet['fillingDate'] = cash_flow_quarter_sheet['fillingDate'].replace('-', '')
                    cash_flow_quarter_sheet['acceptedDate'] = cash_flow_quarter_sheet['acceptedDate'].replace('-', '')
                    cash_flow_quarter_dataframe = pd.DataFrame([cash_flow_quarter_sheet])
                    cash_flow_quarter_dataframe.to_csv(cash_flow_quarter_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')
        cash_flow_annual_name = annual_name + '/' + stock + '--cash_flow_annual_data.csv'
        if (os.path.exists(cash_flow_annual_name)) == False:
            try:
                cash_flow_annual_sheets = get_cash_flow_annual_sheets(stock)
                cash_flow_annual_sheet = cash_flow_annual_sheets[0]
                cash_flow_annual_sheet['date'] = cash_flow_annual_sheet['date'].replace('-', '')
                cash_flow_annual_sheet['fillingDate'] = cash_flow_annual_sheet['fillingDate'].replace('-', '')
                cash_flow_annual_sheet['acceptedDate'] = cash_flow_annual_sheet['acceptedDate'].replace('-', '')
                cash_flow_annual_dataframe = pd.DataFrame([cash_flow_annual_sheet])
                cash_flow_annual_dataframe.to_csv(cash_flow_annual_name, index=False, mode='w')
                for j in range(1, len(cash_flow_annual_sheets)):
                    cash_flow_annual_sheet = cash_flow_annual_sheets[j]
                    cash_flow_annual_sheet['date'] = cash_flow_annual_sheet['date'].replace('-', '')
                    cash_flow_annual_sheet['fillingDate'] = cash_flow_annual_sheet['fillingDate'].replace('-', '')
                    cash_flow_annual_sheet['acceptedDate'] = cash_flow_annual_sheet['acceptedDate'].replace('-', '')
                    cash_flow_annual_dataframe = pd.DataFrame([cash_flow_annual_sheet])
                    cash_flow_annual_dataframe.to_csv(cash_flow_annual_name, index=False, mode='a', header=None)
            except:
                print(stock + ':failed')

if __name__ == "__main__":
    # stock_list = get_specific_country_stock_list('JP')
    # print(stock_list)
    # print(len(stock_list))
    # stock_list = get_specific_exchange_stock_list('')
    # # print(stock_list)
    # print(len(stock_list))
    exchange_suffix = '.T'
    folder_name = 'Japan'
    get_financial_sheets(exchange_suffix, folder_name)

