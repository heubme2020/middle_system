from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.optim import Adam
import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader, random_split
import tools
import get_train_image
import random
import torch.nn.functional as F
from torchvision import models, transforms
import tushare as ts
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
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(15312, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.fc4 = nn.Linear(3, 1)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.csvs_path = tools.get_specified_files(data_path, '.csv')
        random.shuffle(self.csvs_path)

    def __getitem__(self, index):
        #读取data和标签
        csv_path = self.csvs_path[index]
        data = pd.read_csv(csv_path)
        class_name = data['class_label'][0]
        # class_name = np.array([class_name])
        # class_name = class_name.astype(np.int64)
        # label = torch.from_numpy(class_name)
        # one_hot = torch.nn.functional.one_hot(label, num_classes=3)
        data = data.drop(['Unnamed: 0'], axis='columns')
        data = data.drop(['class_label'], axis='columns')
        data = data.values
        data = data.reshape(1, 128*144)
        data_tensor = torch.from_numpy(data.values)
        data_tensor = data_tensor.reshape(1, 128, 144)
        class_tensor = torch.from_numpy(np.asarray(class_name))
        # class_tensor = class_tensor.reshape(1, 1)
        # image = image.reshape(3, 128, 128)
        return data_tensor, class_tensor

    def __len__(self):
        return len(self.csvs_path)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs, batch_size):
    # Instantiate a neural network model
    # model = torch.load('vit_b_16_valuation_pt')
    # model = torchvision.models.resnet18(num_classes=3)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model = Net()
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    torch.backends.cudnn.benchmark = True
    model.to(device)
    #加载数据集
    train_dataset = SelfDataSet('middle_system/append')
    train, valid = random_split(train_dataset,[0.7,0.3])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=0.01)
    min_valid_loss = float('inf')

    for e in range(num_epochs):
        train_loss = 0.0
        model.train()  # Optional when not using Model Specific layer
        for data, labels in train_loader:
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            target = model(data)
            # label_squeeze = torch.squeeze(labels, dim=1)
            # loss = loss_fn(target, label_squeeze)
            loss = loss_fn(target, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in valid_loader:
            data = data.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            target = model(data)
            label_squeeze = torch.squeeze(labels, dim=1)
            loss = loss_fn(target, label_squeeze)
            valid_loss = loss.item() * data.size(0)

        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model, 'predict_model.pt')

def random_append_samples():
    # 随机获取1995以后的训练数据
    date_today = tools.get_date_today()
    end_date = tools.get_delta_date(date_today, -183)
    trade_date_list = get_train_image.get_trade_date_list_period('19950101', end_date)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load('resnet18.pt')
    ok_count = 0
    fail_count = 0
    while True:
        print('*********************')
        try:
            i = random.randint(0, len(trade_date_list))
            trade_date = trade_date_list[i]
            print(trade_date)
            ts_code_list = get_train_image.get_ts_code_list(trade_date)
            j = random.randint(0, len(ts_code_list))
            ts_code = ts_code_list[j]
            append_image_name = append_folder + '//' + ts_code + '_' + trade_date + '_out.png'
            #求取本季度和下一个季度的最大和最小收盘价
            start_date, _ = tools.get_season_border(trade_date, 0)
            _, end_date = tools.get_season_border(trade_date, 1)
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            # print(daily_data)
            max_price = daily_data['close'].max()
            idmax = daily_data['close'].idxmax()
            min_price = daily_data['close'].min()
            idmin = daily_data['close'].idxmin()

            class_label = 0
            if idmin <= idmax:
                class_label = 0
            if idmin > idmax:
                if (max_price-min_price)/min_price < 1.0:
                    class_label = 1
                else:
                    class_label = 2

            if os.path.exists(append_image_name):
                pass
            else:
                image_src = get_train_image.get_valuation_image(ts_code, trade_date)
                tran = transforms.ToTensor()
                image = tran(image_src)
                image = image.to(device=device, dtype=torch.float32)
                image = image.view(1, 3, 128 ,128)
                out = net(image)
                out = F.softmax(out, dim=1)
                out = out.cpu()
                out = out.detach().numpy()
                out = out[0]
                value_predict = 0.0
                for i in range(3):
                    value_predict = value_predict + float(i)*out[i]
                print(class_label)
                print(out)
                # delta = abs((value_predict-value_estimate)/value_estimate)
                # if delta > 0.2:
                #     append_image_name = append_image_name.replace('.png', '_' + str(int(price_estimate)) + '.png')
                #     cv2.imwrite(append_image_name, image_src)
                #     fail_count = fail_count + 1
                # else:
                #     ok_count = ok_count + 1
                # print('fail:' + str(fail_count))
                # print('ok:' + str(ok_count))

        except:
            pass



def reshape_train_data(folder):
    csv_files = tools.get_specified_files(folder, '.csv')
    data_list = []
    for i in range(len(csv_files)):
        print(float(i)/len(csv_files))
        csv_file = csv_files[i]
        data = pd.read_csv(csv_file)
        data = data.drop(['Unnamed: 0'], axis='columns')
        label = data['class_label'][0]
        data = data.drop(['class_label'], axis='columns')
        data = data.values
        data = data.reshape(1, 128*144)
        data = pd.DataFrame(data)
        data['target'] = label
        data_list.append(data)
    random.shuffle(data_list)
    train_data = pd.concat(data_list, axis=0)
    train_data = train_data.reset_index(drop=True)
    print(train_data)
    train_data.to_csv('predict_train_data.csv', mode='w')

def train_lgb():
    data = pd.read_csv('predict_train_data.csv')
    data = data.drop(['Unnamed: 0'], axis='columns')
    print(data)
    y = data.target
    x = data.drop('target', axis='columns')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    print(x_train)
    x_test = scaler.transform(x_test)
    dump(scaler, 'predict_std_scaler.bin', compress=True)
    ## 定义 LightGBM 模型
    clf = LGBMClassifier(feature_fraction=0.5, learning_rate=0.1, max_depth=3, num_leaves=16)
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
    # 定义参数取值范围
    learning_rate = [0.1, 0.3, 0.6]
    feature_fraction = [0.5, 0.8, 1]
    num_leaves = [16, 32, 64]
    max_depth = [-1, 3, 5, 8]

    parameters = {'learning_rate': learning_rate,
                  'feature_fraction': feature_fraction,
                  'num_leaves': num_leaves,
                  'max_depth': max_depth}
    model = LGBMClassifier(n_estimators=50)

    ## 进行网格搜索
    clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)
    clf = clf.fit(x_train, y_train)
    print(clf.best_params_)
if __name__ == "__main__":
    # Let's build our model
    # train(num_epochs=2000, batch_size=32)
    # random_append_samples()
    # train_dataset = SelfDataSet('middle_system/append')
    # train, valid = random_split(train_dataset,[0.7,0.3])
    # train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = torch.load('predict_model.pt')
    # csv_files = tools.get_specified_files('middle_system/train/2', '.csv')
    # for i in range(len(csv_files)):
    #     csv_file = csv_files[i]
    #     data = pd.read_csv(csv_file)
    #     data = data.drop(['Unnamed: 0'], axis='columns')
    #     data_tensor = torch.from_numpy(data.values)
    #     data_tensor = data_tensor.reshape(1, 128, 144)
    #     data_tensor = data_tensor.to(device=device, dtype=torch.float32)
    #     data_tensor = data_tensor.view(1, 1, 128, 144)
    #     out = net(data_tensor)
    #     out = F.softmax(out, dim=1)
    #     print(out)
    # reshape_train_data('middle_system/append')
    train_lgb()




