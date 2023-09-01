import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import os
from sqlalchemy import create_engine, func, update, delete, MetaData, Table, Column, String
from sqlalchemy.orm import sessionmaker
import csv
from dateutil.relativedelta import relativedelta
import datetime
import shutil
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import warnings
# warnings.filterwarnings('ignore')
#创建数据库
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock')
from torch.utils.data import Dataset, DataLoader


def get_exchange_daily_data(exchange):
    exchange = exchange.lower()
    exchange_folder = 'D:/' + exchange
    if (os.path.exists(exchange_folder)) == False:
        os.mkdir(exchange_folder)
    table_name = 'daily_' + exchange.lower()
    file_name = exchange_folder + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    # return df_csv
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

    table_name = 'indicator_' + exchange.lower()
    file_name = exchange_folder + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    sql = "SELECT * FROM " + table_name
    df = pd.read_sql(sql, session.connection())
    df_csv = df
    # return df_csv
    df_csv.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def date_back(date_str):
    date_str = str(date_str)
    str_list = list(date_str)    # 字符串转list
    str_list.insert(4, '-')  # 在指定位置插入字符串
    str_list.insert(7, '-')  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

def get_exchange_daily_forecast_train_data(exchange):
    exchange = exchange.lower()
    exchange_folder = 'D:/' + exchange
    if (os.path.exists(exchange_folder)) == False:
        os.mkdir(exchange_folder)
    train_data_folder = exchange_folder + '/daily_forecast_train_data'
    if (os.path.exists(train_data_folder)) == False:
        os.mkdir(train_data_folder)

    get_exchange_daily_data(exchange)
    indicator_data = pd.read_csv(exchange_folder + '/indicator_' + exchange + '.csv')
    daily_data = pd.read_csv(exchange_folder + '/daily_' + exchange + '.csv')
    daily_indicator_data = pd.DataFrame()
    daily_indicator_data_name = exchange + '/daily_indicator_' + exchange + '.csv'
    days_input = 512
    days_output = 20

    groups = list(indicator_data.groupby('symbol'))
    for i in range(len(groups)):
        print('train_data_generating:' + str(i/(len(groups) + 1.0)))
        group = groups[i][1]
        group = group.reset_index()
        group.drop(columns=['index'], inplace=True)
        group['endDate'] = group['endDate'].apply(lambda x: date_back(x))
        sheet_count = len(group)
        for j in range(sheet_count-1):
            dates = pd.date_range(start=group['endDate'].iloc[j], end=group['endDate'].iloc[j+1])
            dates = dates.date.tolist()
            for k in range(1, len(dates)-1):
                date = dates[k]
                date_str = date.strftime('%Y-%m-%d')
                data_append = group.iloc[j]
                data_append['endDate'] = date_str
                group = group.append(data_append, ignore_index=True)
        date = group['endDate'].iloc[sheet_count-1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        next_quarter_end = date + relativedelta(months=3)
        next_quarter_end_str = next_quarter_end.strftime('%Y-%m-%d')
        dates = pd.date_range(start=group['endDate'].iloc[sheet_count-1], end=next_quarter_end_str)
        dates = dates.date.tolist()
        for m in range(1, len(dates) - 1):
            date = dates[m]
            date_str = date.strftime('%Y-%m-%d')
            data_append = group.iloc[sheet_count-1]
            data_append['endDate'] = date_str
            group = group.append(data_append, ignore_index=True)
        group['endDate'] = group['endDate'].apply(lambda x: x.replace('-', ''))
        group.sort_values(by='endDate', inplace=True, ascending=True)
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        group = group.rename(columns={'endDate': 'date'})
        group['date'] = group['date'].apply(lambda x: x.replace('-', ''))
        group['date'] = group['date'].astype(int)
        daily_group = daily_data[daily_data['symbol'] == groups[i][0]]
        daily_group.reset_index(inplace=True)
        daily_group.drop(columns='index', inplace=True)
        group = pd.merge(group, daily_group, how='outer', on=['date', 'symbol'])
        group = group.dropna(axis=0, how='any')
        group.reset_index(inplace=True)
        group.drop(columns='index', inplace=True)
        group = group.rename(columns={'numberOfShares': 'totalCapital'})
        group['totalCapital'] = group['totalCapital']*group['close']
        group = group.rename(columns={'dividendPerShare': 'dividendPer'})
        group['dividendPer'] = group['dividendPer']/group['close']
        group = group.rename(columns={'netAssetValuePerShare': 'netAssetValuePer'})
        group['netAssetValuePer'] = group['netAssetValuePer']/group['close']
        group = group.rename(columns={'salesPerShare': 'salesPer'})
        group['salesPer'] = group['salesPer']/group['close']
        group = group.rename(columns={'cashFlowPerShare': 'cashFlowPer'})
        group['cashFlowPer'] = group['cashFlowPer']/group['close']
        group = group.rename(columns={'dcfPerShare': 'dcfPer'})
        group['dcfPer'] = group['dcfPer']/group['close']
        daily_indicator_data = pd.concat([daily_indicator_data, group], axis=0)
        daily_indicator_data.reset_index(inplace=True)
        daily_indicator_data.drop(columns='index', inplace=True)
        group.drop(columns='symbol', inplace=True)
        ##上市3个月以后才可以预测
        if len(group) < 80:
            continue
        else:
            padding_data = pd.concat([group.iloc[[0]]]*(days_input-60))
            group_data = pd.concat([padding_data, group])
            group_data = group_data.reset_index()
            group_data.drop(columns=['index'], inplace=True)
            group_data_length = len(group_data)
            #一花一世界，我们认为用一只股票的所有日期数据做归一化是比较合理的
            col_names = group_data.columns.values
            for k in range(1, len(group_data.columns)):
                mean = group_data[col_names[k]].mean()
                std = group_data[col_names[k]].std()
                group_data[col_names[k]] = (group_data[col_names[k]] - mean) / std
            for j in range(days_input, group_data_length):
                end_index = j + days_output
                if end_index > group_data_length: break
                train_data = group_data.iloc[j-days_input:end_index]
                date = group_data['date'].iloc[j]
                train_data_name = train_data_folder + '/' + groups[i][0] + '_' + str(date) + '.csv'
                if (os.path.exists(train_data_name)) == True:
                    continue
                else:
                    train_data = train_data.reset_index()
                    train_data.drop(columns=['index'], inplace=True)
                    train_data = train_data.fillna(0)
                    train_data.replace([np.inf, -np.inf], 0, inplace=True)
                    train_data.to_csv(train_data_name, index=False)
    # daily_indicator_data.to_csv(daily_indicator_data_name)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], input_sequences[end_ix-1:out_end_ix]
        seq_mean = seq_x.mean(axis=0)
        seq_std = seq_x.std(axis=0)
        seq_x = seq_x.sub(seq_mean, axis=1)
        seq_x = seq_x.div(seq_std, axis=1)
        seq_y = seq_y.sub(seq_mean, axis=1)
        seq_y = seq_y.div(seq_std, axis=1)
        seq_y = seq_y.T
        X.append(seq_x), y.append(seq_y)
    z = list(zip(X, y))
    random.shuffle(z)
    X[:], y[:] = zip(*z)
    return np.array(X), np.array(y)

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
        super(GRU, self).__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the GRU
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each GRU layer
        self.device = device
        # # 加些归一化
        # self.fc0 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # GRU
        # GRU model
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # GRU
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # # cell state
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # propagate input through GRU
        output, hn = self.GRU(x, h_0)  # (input, hidden, and internal state)
        # hn = hn.view(-1, self.hidden_size)
        pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(output), self.fc5(output)
        # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
        pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:, -1, :]
        out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
        return out
    
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
        super(LSTM, self).__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        self.device = device
        # # # 加些归一化
        # self.fc0 = nn.LSTM(input_size, input_size)  # lstm
        # self.norm = nn.LayerNorm(input_size)  # lstm
        # LSTM model
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        # hn = hn.view(-1, self.hidden_size)
        pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(output), self.fc5(output)
        # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
        pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:, -1, :]
        out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
        return out

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(x, (h_0, c_0))
        return output, h, c

class Decoder(nn.Module):
    def __init__(self, input_length, hidden_size, num_layers=1, batch_size=1024, device=None):
        super(Decoder, self).__init__()
        self.input_length = input_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = hidden_size
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        # x = x.view(self.batch_size, 1, self.input_size)
        lstm_out, (hn, cn) = self.lstm(x, (hn, cn))
        pred1, pred2, pred3, pred4, pred5 = self.fc1(hn), self.fc2(hn), self.fc3(hn), self.fc4(hn), self.fc5(hn)
        pred1, pred2, pred3, pred4, pred5 = pred1[-1, :, :], pred2[-1, :, :], pred3[-1, :, :], pred4[-1, :, :], pred5[-1, :, :]
        out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
        return out, hn, cn

class Seq2Seq(nn.Module):
    def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024, device=None):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(self.device)
        self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(self.device)

    def forward(self, x):
        encoder_output, hidden, cell = self.encoder(x)
        batch_length = encoder_output.shape[0]
        encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
        outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
        for day_idx in range(self.output_length):
            print(encoder_output.shape)
            print(hidden.shape)
            out, hidden, cell = self.decoder(encoder_output, hidden, cell)
            outputs[:, :, day_idx] = out[:, :, -1]
        return outputs

class Attention(nn.Module):
    def __init__(self, input_hid_dim, output_hid_dim):
        super().__init__()
        self.attention = nn.Linear(input_hid_dim + output_hid_dim, input_hid_dim)
        self.align = nn.Linear(output_hid_dim, 1)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        feature_size = encoder_outputs.shape[2]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, src_len, 1)
        attention_outputs = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        align_score = self.align(attention_outputs).squeeze(2)
        attention_weights_days = F.softmax(align_score, dim=1).unsqueeze(2)
        attention_weights_days = attention_weights_days.repeat(1, 1, feature_size)
        attention_weight_features = F.softmax(attention_outputs, dim=2)
        attention_weights = attention_weight_features.mul(attention_weights_days)
        return attention_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024, device=None):
        super(Seq2SeqWithAttention, self).__init__()
        self.input_size = input_size
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(self.device)
        self.attention = Attention(self.input_size, self.hidden_size).to(self.device)
        self.vector = nn.Linear(input_size + input_size, input_size)
        self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(self.device)

    def forward(self, x):
        encoder_output, hidden, cell = self.encoder(x)
        batch_length = encoder_output.shape[0]
        encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
        outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
        for day_idx in range(self.output_length):
            # print(encoder_output.shape)
            # print(hidden.shape)
            attention_weights = self.attention(hidden, encoder_output)
            vector_output = self.vector(torch.cat((encoder_output, attention_weights), dim=2))
            out, hidden, cell = self.decoder(vector_output, hidden, cell)
            outputs[:, :, day_idx] = out[:, :, -1]
        return outputs

def transform_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num

        self.W_Q = nn.Linear(self.D, self.D*self.H)
        self.W_K = nn.Linear(self.D, self.D*self.H)
        self.W_V = nn.Linear(self.D, self.D*self.H)
        self.W_O = nn.Linear(self.D*self.H, self.D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x):
        Q = self.W_Q(x)  # (B, S, D)
        K = self.W_K(x)  # (B, S, D)
        V = self.W_V(x)  # (B, S, D)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_weights, V)  # (B, H, S, D)
        output = self.concat_heads(output) # (B, S, D*H)
        output = self.W_O(output)

        return output


class Transformer_Encoder(nn.Module):
    def __init__(self, D, H, device=None):
        super(Transformer_Encoder, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num
        self.device = device

        self.attention = MultiHeadAttention(D, H)
        self.norm1 = nn.LayerNorm(self.D)
        self.feedforward = nn.Sequential(
            nn.Linear(self.D, 4 * self.D),
            nn.ReLU(),
            nn.Linear(4 * self.D, self.D)
        )
        self.norm2 = nn.LayerNorm(self.D)

    def forward(self, x):
        attention_output = self.attention(x)
        attention_output = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(attention_output)
        output = self.norm2(attention_output + feedforward_output)
        return output

class Transformer_Decoder(nn.Module):
    def __init__(self, D, S, device=None):
        super(Transformer_Decoder, self).__init__()
        self.D = D  # feature num
        self.S = S  # seq_length
        self.device = device

        self.next1 = nn.Linear(self.D*2, self.D)
        self.next2 = nn.Linear(self.D*self.S, self.D)

    def forward(self, x, encoder_output):
        decoder_input = torch.cat((x, encoder_output), dim=2)
        decoder_output = self.next1(decoder_input)
        B, _, _ = decoder_input.shape
        decoder_output = decoder_output.reshape(B, self.S*self.D)
        decoder_output = self.next2(decoder_output)
        return decoder_output

class Transformer(nn.Module):
    def __init__(self, input_dim, seq_length, num_heads, out_length, device=None):
        super(Transformer, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.H = num_heads
        self.O = out_length
        self.device = device

        self.encoder = Transformer_Encoder(self.D, self.H, self.device).to(self.device)
        self.decoder = Transformer_Decoder(self.D, self.S, self.device).to(self.device)
        self.output = nn.Linear(input_dim, 5)

    def forward(self, x):
        encoder_output = self.encoder(x)
        batch_length = encoder_output.shape[0]
        outputs = torch.zeros(5, batch_length, self.O).to(self.device)
        for day_idx in range(self.O):
            decoder_output = self.decoder(x, encoder_output)
            # print(decoder_output.shape)
            output = self.output(decoder_output)
            output = output.permute(1, 0)
            outputs[:, :, day_idx] = output
            decoder_output = torch.unsqueeze(decoder_output, dim=1)
            # print(x.shape)
            # print(decoder_output.shape)
            x = torch.cat((x, decoder_output), dim=1)
            x = x[:, 1:, :]
        return outputs

class CSV_Dataset(Dataset):
    def __init__(self, csv_files, device=None):
        self.csv_files = csv_files
        self.device = device
    def __len__(self):
        return len(self.csv_files)
    def __getitem__(self, idx):
        file = self.csv_files[idx]
        data = pd.read_csv(file)
        input_data = data.iloc[:-20]
        input_data = input_data.iloc[:, 1:]
        fact_data = data.iloc[-20:]
        fact_data = fact_data.iloc[:, -5:]

        if self.device is None:
            input_data = input_data.values
            fact_data = fact_data.values
        else:
            input_data = input_data.values
            fact_data = fact_data.values
            input_data = torch.tensor(input_data, device=self.device)
            fact_data = torch.tensor(fact_data, device=self.device)
        return input_data, fact_data

def median_squared_error(output, target):
    output_median = torch.median(output, dim=1).values
    target_median = torch.median(target, dim=1).values
    loss = torch.mean(((output_median/target_median - 1.0) ** 2))
    return loss

def max_squared_error(output, target):
    output_max = torch.max(output, dim=1).values
    target_max = torch.max(target, dim=1).values
    loss = torch.mean(((output_max/target_max - 1.0) ** 2))
    return loss

def training_loop(n_epochs, model, optimiser, loss_fn, train_loader, test_loader, model_name):
    # model = model.float()
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            # model.train()
            input_data, fact_data = data
            input_data = input_data.float()
            fact_data = fact_data.float()
            output_data = model(input_data)
            optimiser.zero_grad() # calculate the gradient, manually setting to 0
            loss = 0
            ratio = [0.2, 0.2, 0.2, 0.2, 0.2]
            for k in range(5):
                # loss_k = median_squared_error(output_data[k, :, :], fact_data[:, :, k])
                # loss_k = max_squared_error(output_data[k, :, :], fact_data[:, :, k])
                loss_k = loss_fn(output_data[k, :, :], fact_data[:, :, k])
                # print(loss_k)
                loss = loss + loss_k*ratio[k]
            loss.backward() # calculates the loss of the loss function
            optimiser.step() # improve from loss, i.e backprop
            print("Epoch: %d, train loss: %1.5f" % (epoch, loss.item()))
        # model_name = 'daily_forcast.pt'
        torch.save(model, model_name)

def forecast_plt(model_path, exchange):
    device = torch.device(0)
    exchange = exchange.lower()
    model = torch.load(model_path)
    print(model.eval())
    data_name = exchange + '.csv'
    data = pd.read_csv(data_name)
    data = data[['date', 'open', 'low', 'high', 'close', 'volume']]
    test_data = data.iloc[:522]
    test_data = test_data.reset_index()
    test_data.drop(columns=['index'], inplace=True)
    test_data.sort_values(by='date', inplace=True, ascending=True)
    test_data = test_data.reset_index()
    test_data.drop(columns=['index'], inplace=True)
    test_input = test_data.iloc[:512]
    test_output = test_data.iloc[512:]
    print(test_input)
    test_input = test_input[['open', 'low', 'high', 'close', 'volume']]
    mean = test_input.mean(axis=0)
    std = test_input.std(axis=0)
    print(mean)
    print(std)
    test_input = test_input.sub(mean, axis=1)
    print(test_input)
    test_input = test_input.div(std, axis=1)
    back_input = test_input.mul(std, axis=1)
    back_input = back_input.add(mean, axis=1)
    print(back_input)
    print(test_input)
    test_output = test_data.iloc[-10:]
    test_input = test_input.values
    test_input = torch.tensor(test_input)
    test_input = test_input.float()
    test_input = test_input.reshape(1, 512, 5)
    test_input = torch.tensor(test_input, device=device)
    yhat = model(test_input)
    yhat = yhat.cpu()
    yhat = yhat.detach().numpy()
    yhat = pd.DataFrame(yhat[:,0,:])
    # yhat = yhat.values
    print(yhat)
    # print(yhat.shape)
    yhat = yhat.T
    # print(yhat)
    #
    # # yhat = pd.DataFrame(yhat)
    # yhat = yhat.values
    # print(yhat)
    # yhat = pd.DataFrame(yhat)
    yhat.columns = [['open', 'low', 'high', 'close', 'volume']]
    yhat.reset_index(inplace=True)
    yhat.drop(columns=['index'], inplace=True)
    print(yhat)
    # # yhat = yhat.astype('float64')
    # # print(yhat.info())
    print(std)
    yhat['open'] = yhat['open']*std['open']
    yhat['low'] = yhat['low']*std['low']
    yhat['high'] = yhat['high']*std['high']
    yhat['close'] = yhat['close']*std['close']
    yhat['volume'] = yhat['volume']*std['volume']
    yhat['open'] = yhat['open'] + mean['open']
    yhat['low'] = yhat['low'] + mean['low']
    yhat['high'] = yhat['high'] + mean['high']
    yhat['close'] = yhat['close'] + mean['close']
    yhat['volume'] = yhat['volume'] + mean['volume']
    # yhat = yhat.mul(std, axis=1)
    print(yhat)
    print(test_output)
    # yhat = yhat + mean
    # print(yhat)

def get_csv_files(folder):
    csv_files = []
    for info in os.listdir(folder):
        csv_file = os.path.join(folder, info)
        csv_files.append(csv_file)
    return csv_files

def copy_csv_files(csv_files, ratio=0.2):
    random.shuffle(csv_files)
    copy_num = int(ratio*len(csv_files))
    train_files = csv_files[:copy_num]
    for i in range(len(train_files)):
        print('data copying:' + str(float(i)/float(len(train_files))))
        train_file = train_files[i]
        train_file_name_new = os.path.basename(train_file)
        train_file_name_new = os.path.join('C:/Users/86155/PycharmProjects/Stock/daily_forecast_data/', train_file_name_new)
        shutil.copy(train_file, train_file_name_new)

def train_model():
    device = torch.device(0)
    n_epochs = 10  # 1000 epochs
    learning_rate = 0.01  # 0.001 lr
    input_size = 16  # number of features
    hidden_size = 16  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers
    batch_size = 128
    num_classes = 20  # number of output classes
    input_length = 512
    #
    # model = torch.load('daily_forcast.pt')
    # model = LSTM(num_classes, input_size, hidden_size, num_layers, device)
    # model = Seq2Seq(input_size, input_length, num_classes, hidden_size, num_layers, batch_size, device)
    # model = Seq2SeqWithAttention(input_size, input_length, num_classes, hidden_size, num_layers, batch_size, device)
    model = Transformer(input_size, input_length, 5, num_classes, device)
    model = model.to(device)
    model_name = 'Transformer_daily_forcast.pt'
    # model = model.float()
    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    # loss_fn = torch.nn.L1Loss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    train_folder = 'daily_forecast_data'
    csv_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    print(len(csv_files))
    random.shuffle(csv_files)
    ratio = 0.5
    test_num = int(ratio*len(csv_files))
    train_files = csv_files[:-test_num]
    test_files = csv_files[-test_num:]
    train_dataset = CSV_Dataset(train_files, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset = CSV_Dataset(test_files, device=device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    training_loop(n_epochs=n_epochs,
                  model=model,
                  optimiser=optimiser,
                  loss_fn=loss_fn,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  model_name=model_name)

if __name__ == '__main__':
    train_model()
    # forecast_plt('Bitcoin_daily_forcast.pt', 'Bitcoin')
    # train_model()
    # get_exchange_daily_data('Taipei')
    # get_exchange_daily_forecast_train_data('Shenzhen')
    # data = pd.read_csv('Taipei/daily_forecast_train_data/3551.TPEx_20151029.csv')
    # print(data)
    # data = data.iloc[:,1:]
    # print(data)
    # data = data.iloc[:,-5:]
    # print(data)
    # get_exchange_daily_forecast_train_data('Jakarta')
    # csv_files = get_csv_files('D:/shenzhen/daily_forecast_train_data')
    # print(len(csv_files))
    # copy_csv_files(csv_files, 0.1)
