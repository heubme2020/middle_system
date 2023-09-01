import torch.nn as nn
import torchvision
import numpy as np
from torch.optim import Adam
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import tools
import get_train_image
import random
import torch.nn.functional as F
from torchvision import models, transforms
import tushare as ts
ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')


class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))


    def __getitem__(self, index):
        #读取图片和标签
        image_path = self.imgs_path[index]
        class_name = image_path.split('_')
        class_name = class_name[-1]
        class_name = class_name.split('.')
        class_name = int(class_name[0])
        class_name = np.array([class_name])
        class_name = class_name.astype(np.int64)
        label = torch.from_numpy(class_name)
        one_hot = torch.nn.functional.one_hot(label, num_classes=3)

        image = cv2.imread(image_path)
        image = image.reshape(3, 128, 128)
        return image, one_hot

    def __len__(self):
        return len(self.imgs_path)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs, batch_size):
    # Instantiate a neural network model
    # model = torch.load('vit_b_16_valuation_pt')
    model = torchvision.models.resnet18(num_classes=3)
    print(model)
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    torch.backends.cudnn.benchmark = True
    model.to(device)
    #加载数据集
    train_dataset = SelfDataSet('valuation')
    train, valid = random_split(train_dataset,[0.7,0.3])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
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
            label_squeeze = torch.squeeze(labels, dim=1)
            loss = loss_fn(target, label_squeeze)
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
            torch.save(model, 'resnet18' + '.pt')

    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     i = 0
    #     for image, label in train_loader:
    #         optimizer.zero_grad(set_to_none=True)
    #         image = image.to(device=device, dtype=torch.float32)
    #         label = label.to(device=device, dtype=torch.float32)
    #         pred=model(image)
    #         label_squeeze = torch.squeeze(label, dim=1)
    #         loss = loss_fn(pred, label_squeeze)
    #         loss.backward()
    #         i = i + 1
    #         running_loss = running_loss+loss.item()
    #         optimizer.step()
    #     loss_avg_epoch = running_loss/i
    #     print('epoch: %d avg loss: %f' % (epoch, loss_avg_epoch))
    #     if loss_avg_epoch < bes_los:
    #         bes_los = loss_avg_epoch
    #         torch.save(model, 'resnet18_valuation_pt')


def random_append_samples():
    append_folder = 'append'
    if (os.path.exists(append_folder)) == False:
        os.mkdir(append_folder)
    append_folder = os.getcwd() + '//' + 'append'
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
if __name__ == "__main__":
    # Let's build our model
    # train(num_epochs=2000, batch_size=32)
    random_append_samples()
