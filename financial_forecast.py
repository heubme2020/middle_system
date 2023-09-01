import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
from financial_models import Decoder, Decoder_One, Transformer, Transformer_One, TransformerL_One, TransformerM_One
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

e_power = 1.0/math.e

class Dataset(Dataset):
    def __init__(self, h5_files, device=None):
        self.h5_files = h5_files
        self.device = device

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        file = self.h5_files[idx]
        data = pd.read_hdf(file)
        input_data = data.iloc[:-256]
        input_data = input_data.iloc[:, 1:]
        # 不用-256，而用-257多一天就是为了删除第一个重复的
        fore_data = data.iloc[-257:]
        fact_data = fore_data.iloc[:, 1:3].join(fore_data.iloc[:, 15:-12])
        fact_data = fact_data.drop_duplicates(subset=['numberOfShares', 'dividend', 'revenue','costOfRevenue',
                                                      'capitalExpenditure', 'freeCashFlow'], keep='first')
        fact_data = fact_data.reset_index()
        fact_data.drop(columns=['index'], inplace=True)
        fact_data = fact_data.iloc[1:2]
        # fact_data = fact_data.iloc[1:5]
        if self.device is None:
            input_data = input_data.values
            fact_data = fact_data.values
        else:
            input_data = input_data.values
            fact_data = fact_data.values
            input_data = torch.tensor(input_data, device=self.device)
            fact_data = torch.tensor(fact_data, device=self.device)
        # print(input_data.shape)
        # print(fact_data.shape)
        return input_data, fact_data

def weight_mae_loss(output, target):
    output = torch.squeeze(output)
    # print(output.shape)
    target = torch.squeeze(target)
    loss = torch.mean(abs(output[:] - target[:]))
    # loss_0 = torch.mean(0.4 * abs(output[:, 0] - target[:, 0]))
    # loss_1 = torch.mean(0.3 * abs(output[:, 1] - target[:, 1]))
    # loss_2 = torch.mean(0.2 * abs(output[:, 2] - target[:, 2]))
    # loss_3 = torch.mean(0.1 * abs(output[:, 3] - target[:, 3]))
    # loss = loss_0 + loss_1 + loss_2 + loss_3
    return loss

def weight_mse_loss(output, target):
    output = torch.squeeze(output)
    # print(output.shape)
    target = torch.squeeze(target)
    loss = torch.mean((output[:] - target[:])**2)
    # loss_0 = torch.mean(0.4 * abs(output[:, 0] - target[:, 0]))
    # loss_1 = torch.mean(0.3 * abs(output[:, 1] - target[:, 1]))
    # loss_2 = torch.mean(0.2 * abs(output[:, 2] - target[:, 2]))
    # loss_3 = torch.mean(0.1 * abs(output[:, 3] - target[:, 3]))
    # loss = loss_0 + loss_1 + loss_2 + loss_3
    return loss

def training_loop(n_epochs, model, optimiser, train_loader, validate_loader, model_name):
    feature_importance = pd.read_csv('feature_importance.csv')
    validate_length = len(validate_loader)
    min_validate_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.2, patience=5, verbose=True)
    for epoch in range(n_epochs):
        mean_loss = 0
        step_num = 0
        for i, data in enumerate(train_loader):
            input_data, fact_data = data
            input_data = input_data.float()
            fact_data = fact_data.float()
            output_data = model(input_data)
            # print(output_data.shape)
            optimiser.zero_grad()  # calculate the gradient, manually setting to 0
            loss = 0
            for k in range(fact_data.shape[2]):
                loss_k = weight_mae_loss(output_data[:, :, k], fact_data[:, :, k])
                loss = loss + loss_k*feature_importance.iloc[0, k]
            mean_loss = (mean_loss*step_num + loss)/float(step_num + 1)
            step_num = step_num + 1
            loss.backward()  # calculates the loss of the loss function
            optimiser.step()  # improve from loss, i.e backprop
            print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min loss: %1.5f" % (epoch, loss.item(), mean_loss,
                                                                                       min_validate_loss))
        # 验证集部分
        total_validate_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validate_loader):
                input_data, fact_data = data
                input_data = input_data.float()
                fact_data = fact_data.float()
                output_data = model(input_data)
                loss = 0
                for k in range(fact_data.shape[2]):
                    loss_k = weight_mae_loss(output_data[:, :, k], fact_data[:, :, k])
                    loss = loss + loss_k*feature_importance.iloc[0, k]
                total_validate_loss = total_validate_loss + loss
        total_validate_loss = total_validate_loss/float(validate_length)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, total_validate_loss))
        if total_validate_loss < min_validate_loss:
            min_validate_loss = total_validate_loss
            torch.save(model, model_name)
        scheduler.step(total_validate_loss)  # 在每个epoch结束时调用学习率调度器

def train_model():
    device = torch.device(0)
    n_epochs = 100  # 1000 epochs
    learning_rate = 0.01  # 0.001 lr
    input_size = 125  # number of features
    batch_size = 16
    input_length = 768
    model_name = 'transformerm_one_financial_forcast.pt'
    model = TransformerM_One(input_size, input_length, device)
    if os.path.exists(model_name):
        model = torch.load(model_name)
    model = model.to(device)
    # model = model.float()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_folder = 'financial_daily_data/train'
    # clean_data(train_folder)
    h5_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]
    print(len(h5_files))
    random.seed(1024)
    random.shuffle(h5_files)
    ratio = 0.2
    validate_num = int(ratio * len(h5_files))
    train_files = h5_files[:-validate_num]
    validate_files = h5_files[-validate_num:]
    train_dataset = Dataset(train_files, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, persistent_workers=True, prefetch_factor=3)
    validate_dataset = Dataset(validate_files, device=device)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size)

    training_loop(n_epochs=n_epochs,
                  model=model,
                  optimiser=optimiser,
                  train_loader=train_loader,
                  validate_loader=validate_loader,
                  model_name=model_name)

def metric_model(model_path, test_folder):
    feature_importance = pd.read_csv('feature_importance.csv')
    # feature_importance = feature_importance.iloc[:, 2:]
    device = torch.device(0)
    model = torch.load(model_path)
    print(model.eval())
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.h5')]
    mae_mean = 0
    mae_baseline_mean = 0
    k = 0
    for i in tqdm(range(len(test_files))):
        test_file = test_files[i]
        data = pd.read_hdf(test_file)
        input_data = data.iloc[:-256]
        input_data = input_data.iloc[:, 1:]
        input_data = input_data.values
        input_data = torch.tensor(input_data, device=device)
        input_data = input_data.float()
        input_data = input_data.reshape(1, 768, 125)
        output_data = model(input_data)
        output_data = output_data.cpu()
        output_data = output_data.detach().numpy()
        output_data = pd.DataFrame(output_data[0,:,:])
        fore_data = data.iloc[-257:]
        fact_data = fore_data.iloc[:, 1:3].join(fore_data.iloc[:, 15:-12])
        fact_data = fact_data.drop_duplicates(subset=['numberOfShares', 'dividend', 'revenue','costOfRevenue',
                                                      'capitalExpenditure', 'freeCashFlow'], keep='first')
        fact_data = fact_data.reset_index()
        fact_data.drop(columns=['index'], inplace=True)
        last_data = fact_data.iloc[0:1]
        fact_data = fact_data.iloc[1:2]
        fact_data = fact_data.reset_index(drop=True)
        output_data.columns = fact_data.columns
        mae_data = abs(fact_data - output_data)
        mae_baseline = abs(fact_data - last_data)
        mae_data *= feature_importance.iloc[0]
        mae_baseline *= feature_importance.iloc[0]
        mae_sum_data = mae_data.sum(axis=1)
        mae_sum_baseline = mae_baseline.sum(axis=1)
        # print(mae_sum_data)
        # mae = mae_sum_data[0]*0.4 + mae_sum_data[1]*0.3 + mae_sum_data[2]*0.2 + mae_sum_data[3]*0.1
        mae_mean = (mae_mean*k + mae_sum_data[0])/float(k+1)
        mae_baseline_mean = (mae_baseline_mean*k + mae_sum_baseline[0])/float(k+1)
        k = k + 1
        # print(mae)
    print(mae_mean)
    print(mae_baseline_mean)

def prepare_hdf5_data(folder):
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    print(len(csv_files))
    for i in tqdm(range(len(csv_files))):
        csv_file = csv_files[i]
        hdf_file = csv_file.replace('.csv', '.h5')
        data = pd.read_csv(csv_file)
        data.to_hdf(hdf_file, key='data', mode='w')
        os.remove(csv_file)

if __name__ == '__main__':
    # train_model()
    metric_model('transformerm_one_financial_forcast.pt', 'financial_daily_data/test')
    # prepare_hdf5_data('financial_daily_data/test')