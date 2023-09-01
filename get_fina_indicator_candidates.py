import pandas as pd
import tushare as ts
import tushare_tools as tt
from tushare_tools import fina_indicator_param_list
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import random
import time
import os
import tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import joblib
from joblib import dump, load
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn.functional as F

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')


# 建立神经网络
class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

def get_fina_indicator_candidate_stocks_hist_data():
    fina_indicator_folder = 'fina_indicator'
    if (os.path.exists(fina_indicator_folder)) == False:
        os.mkdir(fina_indicator_folder)
    fina_indicator_folder = os.getcwd() + '//' + 'fina_indicator'
    # 随机获取1995以后的训练数据
    pre_year = str(int(tools.get_toyear()) - 1)
    last_date = pre_year+'1231'
    ts_code_list = tt.get_ts_code_list(pre_year + '1231')
    for i in range(0, ts_code_list.shape[0]):
        print(float(i)/ts_code_list.shape[0])
        ts_code = ts_code_list[i]
        fina_indicator_data = pro.fina_indicator(ts_code=ts_code)
        fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
        fina_indicator_data = fina_indicator_data.reset_index(drop=True)
        fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['ann_date']>last_date].index)
        fina_indicator_data = fina_indicator_data.reset_index(drop=True)
        fina_indicator_data = fina_indicator_data.fillna(0.0)
        for j in range(fina_indicator_data.shape[0]-2):
            if (fina_indicator_data['grossprofit_margin'][j] > fina_indicator_data['grossprofit_margin'][j+1]) and (fina_indicator_data['q_sales_yoy'][j] > fina_indicator_data['q_sales_yoy'][j+1]) and (fina_indicator_data['dt_eps'][j] > 0) and (fina_indicator_data['q_dt_roe'][j] > fina_indicator_data['q_dt_roe'][j+1]):
                select_data = fina_indicator_data[j:j+3]
                select_data = select_data.reset_index(drop=True)
                select_data_name = fina_indicator_folder + '//' + ts_code + '_' + fina_indicator_data['end_date'][j] + '.csv'
                select_data.to_csv(select_data_name, index=False)
        # print(fina_indicator_data)

def label_candidate_stocks_data(folder):
    csv_list = tools.get_specified_files(folder, 'csv')
    folder_0 = 'fina_indicator/0'
    if (os.path.exists(folder_0)) == False:
        os.mkdir(folder_0)
    folder_1 = 'fina_indicator/1'
    if (os.path.exists(folder_1)) == False:
        os.mkdir(folder_1)
    folder_2 = 'fina_indicator/2'
    if (os.path.exists(folder_2)) == False:
        os.mkdir(folder_2)
    for i in range(len(csv_list)):
        print(float(i)/len(csv_list))
        try:
            csv_file = csv_list[i]
            fina_indicator_data = pd.read_csv(csv_file)
            ts_code = fina_indicator_data['ts_code'][0]
            end_date = str(fina_indicator_data['end_date'][0])
            #求取本季度和上个季度的最大收盘价格
            start_date, _ = tools.get_season_border(end_date, -1)
            _, end_date = tools.get_season_border(end_date, 0)
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            max_price_pre = daily_data['close'].max()
            #求取下两个季度收盘价格的中位值，最大值，最小值
            start_date, _ = tools.get_season_border(end_date, 1)
            _, end_date = tools.get_season_border(end_date, 2)
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            max_price = daily_data['close'].max()
            min_price = daily_data['close'].min()
            median_price = daily_data['close'].median()
            idmax = daily_data['close'].idxmax()
            idmin = daily_data['close'].idxmin()
            # print(daily_data)
            # print(max_price)
            # print(idmax)
            # print(min_price)
            # print(idmin)
            if (median_price/max_price_pre > 1.2) and (max_price/min_price > 1.5) and (idmax < idmin):
                csv_file_new = csv_file.replace('fina_indicator', 'fina_indicator/2')
                os.rename(csv_file, csv_file_new)
            elif (median_price > max_price_pre) and (idmax < idmin):
                csv_file_new = csv_file.replace('fina_indicator', 'fina_indicator/1')
                os.rename(csv_file, csv_file_new)
            else:
                csv_file_new = csv_file.replace('fina_indicator', 'fina_indicator/0')
                os.rename(csv_file, csv_file_new)
            # # input()
        except:
            pass
def load_data(folder, target):
    fina_indicator_mean_std = pd.read_csv('fina_indicator_mean_std.csv')
    csv_files = tools.get_specified_files(folder, '.csv')
    param_num = len(fina_indicator_param_list)
    columns_list = []
    for i in range(param_num*3):
        columns_list.append(str(i))
    fina_indicator_data = []
    for i in range(len(csv_files)):
        csv_file = csv_files[i]
        data_frame = pd.read_csv(csv_file)
        fina_indicator_frame = pd.DataFrame([[0.0] * len(columns_list)], columns=columns_list)
        for j in range(param_num):
            param = fina_indicator_param_list[j]
            fina_indicator_frame[str(j)][0] = data_frame[param][0]
            fina_indicator_frame[str(j+param_num)][0] = data_frame[param][0] - data_frame[param][1]
            fina_indicator_frame[str(j+2*param_num)][0] = data_frame[param][0] + data_frame[param][2] - 2.0*data_frame[param][1]
        fina_indicator_data.append(fina_indicator_frame)
    data = pd.concat(fina_indicator_data, axis=0, ignore_index=True)
    data['target'] = target
    print(data)
    return data
def train_svm(folder):
    # data0 = load_data(folder + '/0', 0)
    # data1 = load_data(folder + '/1', 1)
    # data2 = load_data(folder + '/2', 2)
    # data = pd.concat([data0, data1, data2])
    # data = shuffle(data)
    # data = data.reset_index(drop=True)
    # print(data)
    # data.to_csv('svm_data.csv')
    data = pd.read_csv('svm_data.csv')
    y = data.target
    x = data.drop('target', axis='columns')
    scaler = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    print(pearsonr(x_train, y_train))
    # scaler.fit_transform(x_train)
    # scaler.transform(x_test)
    # selector = SelectKBest(f_classif, k=26)
    # selector.fit(x_train, y_train)
    # print(selector.get_support(True))
    # x_train_new = selector.transform(x_train)
    # x_test_new = selector.transform(x_test)
    # # x_train_new = SelectKBest(chi2, k=9).fit_transform(x_train, y_train)
    # # # rbf_model = SVC(kernel='rbf')
    # # # rbf_model.fit(x_train, y_train)
    # # # print(rbf_model.score(x_test, y_test))
    # linear_model = SVC(kernel='linear')
    # linear_model.fit(x_train_new, y_train)
    # print(linear_model.score(x_test_new, y_test))
    # mlp_model = MLPClassifier(solver='adam', hidden_layer_sizes=(32, 8))
    # mlp_model.fit(x_train_new, y_train)
    # print(mlp_model.score(x_test_new, y_test))
def train_mlp(folder):
    data = pd.read_csv('middle_valuation_data.csv')
    x_train, x_test, y_train, y_test = train_test_split(data.drop('target', axis='columns'), data.target, test_size=0.3)
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 3))
    mlp_model.fit(x_train, y_train)
    print(mlp_model.score(x_test, y_test))

def train_lgb(folder):
    data = pd.read_csv('svm_data.csv')
    data = data.drop('Unnamed: 0', axis='columns')
    print(data)
    y = data.target
    x = data.drop('target', axis='columns')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    dump(scaler, 'std_scaler.bin', compress=True)
    # # 创建成lgb特征的数据集格式
    # lgb_train = lgb.Dataset(x_train, y_train)
    # lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    #
    # # 将参数写成字典下形式
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',  # 设置提升类型
    #     'objective': 'regression',  # 目标函数
    #     'metric': {'l2', 'auc'},  # 评估函数
    #     'num_leaves': 31,  # 叶子节点数
    #     'learning_rate': 0.05,  # 学习速率
    #     'feature_fraction': 0.9,  # 建树的特征选择比例
    #     'bagging_fraction': 0.8,  # 建树的样本采样比例
    #     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    #     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    #     }
    # # 训练 cv and train
    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    # # 保存模型到文件
    # gbm.save_model('lgb_model.txt')
    # # 预测数据集
    # y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # # 评估模型
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # # 创建模型，训练模型
    # gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.1, n_estimators=40)
    # gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
    # # 测试机预测
    # y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
    # # 模型评估
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # # feature importances
    # print('Feature importances:', list(gbm.feature_importances_))
    # # 网格搜索，参数优化
    # estimator = lgb.LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(x_train, y_train)
    # print('Best parameters found by grid search are:', gbm.best_params_)
    ## 定义 LightGBM 模型
    clf = LGBMClassifier(feature_fraction=0.5, learning_rate=0.1, max_depth=3, num_leaves=16)
    # 在训练集上训练LightGBM模型
    clf.fit(x_train, y_train)
    # 模型存储
    joblib.dump(clf, 'lgb_model.pkl')
    ## 在训练集和测试集上分布利用训练好的模型进行预测
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

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
def train_net():
    net = Net(n_feature=64, n_hidden=10, n_output=3)  # 几个类别就几个 output
    pass

def feature_selection():
    data = pd.read_csv('middle_valuation_data.csv')
    print(data)
    select = VarianceThreshold(1)
    select.fit(data)
    select.transform(data)
    index_list = select.get_support(True)
    print(index_list)
    print(len(index_list))
def first_filter():
    fina_indicator_data = pd.read_csv('current_season_fina_indicator_data.csv')
    fina_indicator_data = fina_indicator_data.drop('Unnamed: 0', axis='columns')
    #加载分类模型
    lgb = joblib.load('lgb_model.pkl')
    #加载标准化数据的参数
    scaler = load('std_scaler.bin')
    #初始化
    param_num = len(fina_indicator_param_list)
    columns_list = []
    for i in range(param_num * 3):
        columns_list.append(str(i))

    #推荐股票计数
    count = 0
    ts_code_now = fina_indicator_data['ts_code'][0]
    for i in range(fina_indicator_data.shape[0]):
        ts_code = fina_indicator_data['ts_code'][i]
        end_date = fina_indicator_data['end_date'][i]
        if ts_code_now != ts_code:
            ts_code_now = ts_code
            if (fina_indicator_data['grossprofit_margin'][i] > fina_indicator_data['grossprofit_margin'][i + 1]) and (
                    fina_indicator_data['q_sales_yoy'][i] > fina_indicator_data['q_sales_yoy'][i + 1]) and (
                    fina_indicator_data['dt_eps'][i] > 0) and (
                    fina_indicator_data['q_dt_roe'][i] > fina_indicator_data['q_dt_roe'][i + 1]) and (fina_indicator_data['grossprofit_margin'][i] > 10) and (fina_indicator_data['roe_dt'][i] > 5) and (fina_indicator_data['ocfps'][i] > 0):
                #生成分类数据
                fina_indicator_frame = pd.DataFrame([[0.0] * len(columns_list)], columns=columns_list)
                for k in range(param_num):
                    param = fina_indicator_param_list[k]
                    fina_indicator_frame[str(k)][0] = fina_indicator_data[param][i]
                    fina_indicator_frame[str(k + param_num)][0] = fina_indicator_data[param][i] - fina_indicator_data[param][i+1]
                    fina_indicator_frame[str(k + 2 * param_num)][0] = fina_indicator_data[param][i] + fina_indicator_data[param][i+2] - 2.0 * fina_indicator_data[param][i+1]
                fina_indicator_frame = scaler.transform(fina_indicator_frame)
                # print(fina_indicator_frame)
                y_pred = lgb.predict(fina_indicator_frame)
                if y_pred[0] == 2:
                    count = count + 1
                    print('********************************')
                    print(count)
                    print(ts_code)
                    print(end_date)



if __name__ == "__main__":
    # Let's build our model
    # train(num_epochs=2000, batch_size=32)
    # get_fina_indicator_candidate_stocks_hist_data()
    # label_candidate_stocks_data('fina_indicator')
    # feature_selection()
    # train_lgb('fina_indicator/train')
    # first_filter()
    print(tt.get_dv_ttm('601919.SH', '20220101'))