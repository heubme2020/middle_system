import pandas as pd
import numpy as np
import tools
import cv2
import datetime
import tushare as ts
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

#按上市时间排序获取股票列表
def get_stock_list():
    # 查询当前所有正常上市交易的股票列表
    data0 = pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date')
    data1 = data0.sort_values(by='list_date', ascending=True)
    return data1

#获取指定股票的历史财务数据生成对应图片
def get_financial_images(ts_code, list_date):
    print(ts_code)
    print(list_date)
    date_now = datetime.datetime.now()
    date_now = int(date_now.strftime('%Y%m%d'))
    start_date = datetime.datetime.strptime(list_date, "%Y%m%d")
    #间隔count天读取财务数据
    count = 365
    end_date = (start_date + datetime.timedelta(days=count)).strftime("%Y%m%d")
    #获取financial_data_list
    financial_data_list = []
    while (date_now - int(end_date)) > 0:
        start_date = start_date.strftime("%Y%m%d")
        financial_data = pro.query('fina_indicator', ts_code=ts_code, start_date=start_date, end_date=end_date)
        financial_data = financial_data.sort_values(by='end_date', ascending=True)
        if financial_data.empty:
            pass
        else:
            for i in range(financial_data.shape[0]):
                financial_data_list.append(financial_data.iloc[i])


        start_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        end_date = (start_date + datetime.timedelta(days=count)).strftime("%Y%m%d")

    #生成财务图片
    #先删除重复的数据
    financial_data_list_new = []
    for i in range(len(financial_data_list)-1):
        financial_data = financial_data_list[i]
        financial_data_next = financial_data_list[i+1]
        if financial_data.end_date != financial_data_next.end_date:
            financial_data_list_new.append(financial_data_list[i])
    financial_data_list_new.append(financial_data_list[-1])
    financial_data_list = financial_data_list_new



    for i in range(len(financial_data_list)):
        financial_data = financial_data_list[i]
        print(financial_data)
        #获取本季度利润表
        income_data = pro.income(ts_code=ts_code, start_date=tools.get_delta_date(financial_data.end_date, -90), end_date=tools.get_delta_date(financial_data.end_date, 7))
        #获取本季度资产负债表
        debet_data = pro.balancesheet(ts_code=ts_code, start_date=tools.get_delta_date(financial_data.end_date, -90), end_date=tools.get_delta_date(financial_data.end_date, 7))
        #获取本季度现金流量表
        cash_flow_data = pro.cashflow(ts_code=ts_code, start_date=tools.get_delta_date(financial_data.end_date, -90), end_date=tools.get_delta_date(financial_data.end_date, 7))
        #
        print(cash_flow_data)
        input()
        #生成一张64*64的图片
        img = np.zeros([64, 64, 1], np.uint8)
        #获取股票代码
        stock_code_str = ts_code.split('.')
        stock_code = stock_code_str[0]
        #前两行存储乱七八糟的数据
        #存股票代码
        stock_number0 = int(stock_code[0])*10 + int(stock_code[1])
        stock_number1 = int(stock_code[2])*10 + int(stock_code[3])
        stock_number2 = int(stock_code[4])*10 + int(stock_code[5])
        img[0][0] = stock_number0*2
        img[0][1] = stock_number1*2
        img[0][2] = stock_number2*2
        #存上市日期
        #存公司注册日期
        #存公司注册地点

        #存公告日期
        ann_date0 = int(financial_data.ann_date[0])*10 + int(financial_data.ann_date[1])
        ann_date1 = int(financial_data.ann_date[2])*10 + int(financial_data.ann_date[3])
        ann_date2 = int(financial_data.ann_date[4])*10 + int(financial_data.ann_date[5])
        ann_date3 = int(financial_data.ann_date[6])*10 + int(financial_data.ann_date[7])
        img[1][0] = ann_date0*2
        img[1][1] = ann_date1*2
        img[1][2] = ann_date2*2
        img[1][3] = ann_date3*2
        #存报告内容截至日期
        end_date0 = int(financial_data.end_date[0])*10 + int(financial_data.end_date[1])
        end_date1 = int(financial_data.end_date[2])*10 + int(financial_data.end_date[3])
        end_date2 = int(financial_data.end_date[4])*10 + int(financial_data.end_date[5])
        end_date3 = int(financial_data.end_date[6])*10 + int(financial_data.end_date[7])
        img[1][4] = end_date0*2
        img[1][5] = end_date1*2
        img[1][6] = end_date2*2
        img[1][7] = end_date3*2

        # img_256 = cv2.resize(img, (256, 256))
        # cv2.namedWindow("IMG")
        # cv2.imshow("IMG", img)
        # cv2.waitKey()

#获得指定日期的输出图像
########输出的是财务指标数据为基础构造的单通道图片，对应的tushare的pro.fina_indicator函数
def get_stock_output_image(ts_code, date):
    # 生成一张16*16的图片
    img = np.zeros([16, 16, 1], np.uint8)
    fina_indicator_data_list = pro.fina_indicator(ts_code=ts_code, start_date=date,
                             end_date=tools.get_delta_date(date, 365))
    fina_indicator_data_list['surplus_rese_ps'] = fina_indicator_data_list['surplus_rese_ps'].fillna(0.0)
    fina_indicator_data_list['fcff'] = fina_indicator_data_list['fcff'].fillna(0.0)
    if fina_indicator_data_list.empty:
        return img
    else:
        #倒数第一个是下一个季度的
        fina_indicator_data = fina_indicator_data_list.iloc[-1]
        #----------------------开始对图片进行赋值
        #基本每股收益
        print(fina_indicator_data.index[3] + ':' + str(fina_indicator_data[3]))
        img[0][0] = int(fina_indicator_data[3]*10 + 128)
        #稀释每股收益
        print(fina_indicator_data.index[4] + ':' + str(fina_indicator_data[4]))
        img[0][1] = int(fina_indicator_data[4]*10 + 128)
        #每股营业总收入
        print(fina_indicator_data.index[5] + ':' + str(fina_indicator_data[5]))
        img[0][2] = int(fina_indicator_data[5]*10 + 16)
        #每股营业收入
        print(fina_indicator_data.index[6] + ':' + str(fina_indicator_data[6]))
        img[0][3] = int(fina_indicator_data[6]*10 + 16)
        #每股资本公积
        print(fina_indicator_data.index[7] + ':' + str(fina_indicator_data[7]))
        img[0][4] = int(fina_indicator_data[7]*5 + 16)
        #每股盈余公积
        print(fina_indicator_data.index[8] + ':' + str(fina_indicator_data[8]))
        img[0][5] = int(fina_indicator_data[8]*10 + 16)
        #每股未分配利润
        print(fina_indicator_data.index[9] + ':' + str(fina_indicator_data[9]))
        img[0][6] = int(fina_indicator_data[9]*5 + 64)
        #非经常性损益
        print(fina_indicator_data.index[10] + ':' + str(fina_indicator_data[10]))
        img[0][7] = int(fina_indicator_data[10]/10000000.0)
        #扣非净利润
        print(fina_indicator_data.index[11] + ':' + str(fina_indicator_data[11]))
        img[0][8] = int(fina_indicator_data[11]/20000000.0 + 64)
        #毛利
        print(fina_indicator_data.index[12] + ':' + str(fina_indicator_data[12]))
        img[0][9] = int(fina_indicator_data[12]/20000000.0 + 16)
        #流动比率
        print(fina_indicator_data.index[13] + ':' + str(fina_indicator_data[13]))
        img[0][10] = int(fina_indicator_data[13]*16)
        #速动比率
        print(fina_indicator_data.index[14] + ':' + str(fina_indicator_data[14]))
        img[0][11] = int(fina_indicator_data[14]*16)
        #现金速动比率
        print(fina_indicator_data.index[15] + ':' + str(fina_indicator_data[15]))
        img[0][12] = int(fina_indicator_data[15]*16)
        #应收账款周转率
        print(fina_indicator_data.index[16] + ':' + str(fina_indicator_data[16]))
        img[0][13] = int(fina_indicator_data[16]*16)
        #流动资产周转率
        print(fina_indicator_data.index[17] + ':' + str(fina_indicator_data[17]))
        img[0][14] = int(fina_indicator_data[17]*16)
        #固定资产周转率
        print(fina_indicator_data.index[18] + ':' + str(fina_indicator_data[18]))
        img[0][15] = int(fina_indicator_data[18]*16)
        #总资产周转率
        print(fina_indicator_data.index[19] + ':' + str(fina_indicator_data[19]))
        img[1][0] = int(fina_indicator_data[19]*32)
        #经营活动净收益
        print(fina_indicator_data.index[20] + ':' + str(fina_indicator_data[20]))
        img[1][1] = int(fina_indicator_data[20]/10000000.0 + 64)
        #息税前利润
        print(fina_indicator_data.index[21] + ':' + str(fina_indicator_data[21]))
        img[1][2] = int(fina_indicator_data[21]/20000000.0 + 16)
        #息税折旧摊销前利润
        print(fina_indicator_data.index[22] + ':' + str(fina_indicator_data[22]))
        img[1][3] = int(fina_indicator_data[22]/20000000.0 + 16)
        #企业自由现金流量
        print(fina_indicator_data.index[23] + ':' + str(fina_indicator_data[23]))
        img[1][4] = int(fina_indicator_data[23]/20000000.0 + 64)
        #股权自由现金流量
        print(fina_indicator_data.index[24] + ':' + str(fina_indicator_data[24]))
        img[1][5] = int(fina_indicator_data[24]/20000000.0 + 64)
        #无息流动负债
        print(fina_indicator_data.index[25] + ':' + str(fina_indicator_data[25]))
        img[1][6] = int(fina_indicator_data[25]/20000000.0 + 16)
        #无息非流动负债
        print(fina_indicator_data.index[26] + ':' + str(fina_indicator_data[26]))
        img[1][7] = int(fina_indicator_data[26]/10000000.0 + 16)
        #带息债务
        print(fina_indicator_data.index[27] + ':' + str(fina_indicator_data[27]))
        img[1][8] = int(fina_indicator_data[27]/10000000.0 + 16)
        #净债务
        print(fina_indicator_data.index[28] + ':' + str(fina_indicator_data[28]))
        img[1][9] = int(fina_indicator_data[28]/10000000.0 + 64)
        #有形资产
        print(fina_indicator_data.index[29] + ':' + str(fina_indicator_data[29]))
        img[1][10] = int(fina_indicator_data[29]/10000000.0 + 16)




#获得指定日期的输入图像和输出图像
def get_stock_image(ts_code, date):
    pass

    
stock_list = get_stock_list()
print(stock_list)

get_stock_output_image(stock_list.iloc[5000].ts_code, '20211001')
# for i in range(data.shape[1]):
#     get_financial_images(data.iloc[i].ts_code, data.iloc[i].list_date)


