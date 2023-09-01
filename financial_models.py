import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Decoder_One(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Decoder_One, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device

        self.linear = nn.Linear(input_dim*seq_length, 125)
        self.norm = nn.LayerNorm(125)
        self.gelu = nn.GELU()
        self.output = nn.Linear(125, 101)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, 1, 101).to(self.device)
        decoder_output = x.reshape(batch_size, self.S * self.D)
        decoder_output = self.linear(decoder_output)
        decoder_output = self.norm(decoder_output)
        decoder_output = self.gelu(decoder_output)
        output = self.output(decoder_output)
        outputs[:, 0, :] = output
        return outputs

class Decoder(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Decoder, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device

        self.linear = nn.Linear(input_dim*seq_length, input_dim*4)
        self.norm = nn.LayerNorm(input_dim*4)
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(input_dim*4, 99)
        self.linear2 = nn.Linear(input_dim*4, 99)
        self.linear3 = nn.Linear(input_dim*4, 99)
        self.linear4 = nn.Linear(input_dim*4, 99)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, 4, 99).to(self.device)
        decoder_output = x.reshape(batch_size, self.S * self.D)
        decoder_output = self.linear(decoder_output)
        decoder_output = self.norm(decoder_output)
        decoder_output = self.gelu(decoder_output)
        output1 = self.linear1(decoder_output)
        outputs[:, 0, :] = output1
        output2 = self.linear2(decoder_output)
        outputs[:, 1, :] = output2
        output3 = self.linear3(decoder_output)
        outputs[:, 2, :] = output3
        output4 = self.linear4(decoder_output)
        outputs[:, 3, :] = output4
        return outputs

class ProbAttention(nn.Module):
    def __init__(self):
        super(ProbAttention, self).__init__()
        self.factor = 1

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = 1./math.sqrt(D)
        scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2,1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num

        self.W_Q = nn.Linear(self.D, self.D * self.H)
        self.W_K = nn.Linear(self.D, self.D * self.H)
        self.W_V = nn.Linear(self.D, self.D * self.H)
        self.W_O = nn.Linear(self.D * self.H, self.D)

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

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_weights, V)  # (B, H, S, D)
        output = self.concat_heads(output)  # (B, S, D*H)
        output = self.W_O(output)
        return output

class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Transformer_Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention_output = self.attention(x)
        output = x + attention_output
        output = self.linear(output)
        output = self.norm(output)
        output = self.gelu(output)
        output = self.out(output)
        return output

class Transformer_MulEncoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Transformer_MulEncoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention_output = self.attention(x)
        output = x + attention_output
        output = self.norm(output)
        output = self.gelu(output)
        output = self.out(output)
        return output

class TransformerM_One(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(TransformerM_One, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device
        self.heads_num = 17
        self.encoder = MultiHeadAttention(input_dim, self.heads_num)
        # self.encoder = Transformer_MulEncoder(input_dim, self.heads_num)
        # self.encoder0 = Transformer_MulEncoder(input_dim, self.heads_num)
        # self.encoder1 = Transformer_MulEncoder(input_dim, self.heads_num)
        # self.encoder2 = Transformer_MulEncoder(input_dim, self.heads_num)
        self.decoder = Decoder_One(input_dim, seq_length, device)

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = x*encoder_output
        # encoder_output = self.encoder0(x)
        # encoder_output = self.encoder1(encoder_output)
        # encoder_output = self.encoder2(encoder_output)
        # encoder_output = x + encoder_output
        # encoder_output = x + encoder_output
        # batch_length = encoder_output.shape[0]
        # encoder_output = encoder_output.reshape(batch_length, self.S * self.D)
        outputs = self.decoder(encoder_output)
        return outputs

class TransformerL_One(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(TransformerL_One, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device
        self.heads_num = 16

        self.encoder0 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder1 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder2 = Transformer_Encoder(input_dim, self.heads_num)
        self.decoder = LDecoder_One(input_dim, seq_length, device)

    def forward(self, x):
        encoder_output = self.encoder0(x)
        encoder_output = self.encoder1(encoder_output)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = x + encoder_output
        # batch_length = encoder_output.shape[0]
        # encoder_output = encoder_output.reshape(batch_length, self.S * self.D)
        outputs = self.decoder(encoder_output)
        return outputs

class Transformer_One(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Transformer_One, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device
        self.heads_num = 16

        self.encoder0 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder1 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder2 = Transformer_Encoder(input_dim, self.heads_num)
        self.decoder = Decoder_One(input_dim, seq_length, device)

    def forward(self, x):
        encoder_output = self.encoder0(x)
        encoder_output = self.encoder1(encoder_output)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = x + encoder_output
        # batch_length = encoder_output.shape[0]
        # encoder_output = encoder_output.reshape(batch_length, self.S * self.D)
        outputs = self.decoder(encoder_output)
        return outputs

class Transformer(nn.Module):
    def __init__(self, input_dim, seq_length, device=None):
        super(Transformer, self).__init__()
        self.D = input_dim
        self.S = seq_length
        self.device = device
        self.heads_num = 16

        self.encoder0 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder1 = Transformer_Encoder(input_dim, self.heads_num)
        self.encoder2 = Transformer_Encoder(input_dim, self.heads_num)
        self.decoder = Decoder(input_dim, seq_length, device)

    def forward(self, x):
        encoder_output = self.encoder0(x)
        encoder_output = self.encoder1(encoder_output)
        encoder_output = self.encoder2(encoder_output)
        encoder_output = x + encoder_output
        batch_length = encoder_output.shape[0]
        encoder_output = encoder_output.reshape(batch_length, self.S * self.D)
        outputs = self.decoder(encoder_output)
        return outputs

# class GRU(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
#         super(GRU, self).__init__()
#         self.num_classes = num_classes  # output size
#         self.num_layers = num_layers  # number of recurrent layers in the GRU
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # neurons in each GRU layer
#         self.device = device
#         # # 加些归一化
#         # self.fc0 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # GRU
#         # GRU model
#         self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
#                           batch_first=True)  # GRU
#         self.fc1 = nn.Linear(hidden_size, num_classes)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         self.fc4 = nn.Linear(hidden_size, num_classes)
#         self.fc5 = nn.Linear(hidden_size, num_classes)
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # hidden state
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # # cell state
#         # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         # propagate input through GRU
#         output, hn = self.GRU(x, h_0)  # (input, hidden, and internal state)
#         # hn = hn.view(-1, self.hidden_size)
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(
#             output), self.fc5(output)
#         # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
#         pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:,
#                                                                                                                 -1, :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out
#
#
# class LSTM(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
#         super(LSTM, self).__init__()
#         self.num_classes = num_classes  # output size
#         self.num_layers = num_layers  # number of recurrent layers in the lstm
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # neurons in each lstm layer
#         self.device = device
#         # # # 加些归一化
#         # self.fc0 = nn.LSTM(input_size, input_size)  # lstm
#         # self.norm = nn.LayerNorm(input_size)  # lstm
#         # LSTM model
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, num_classes)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         self.fc4 = nn.Linear(hidden_size, num_classes)
#         self.fc5 = nn.Linear(hidden_size, num_classes)
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # hidden state
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # cell state
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         # propagate input through LSTM
#         output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
#         # hn = hn.view(-1, self.hidden_size)
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(output), self.fc2(output), self.fc3(output), self.fc4(
#             output), self.fc5(output)
#         # pred1, pred2, pred3, pred4, pred5 = self.re1(pred1), self.re2(pred2), self.re3(pred3), self.re4(pred4), self.re5(pred5)
#         pred1, pred2, pred3, pred4, pred5 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :], pred4[:, -1, :], pred5[:,
#                                                                                                                 -1, :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out
#
#
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, batch_size, device=None):
#         super(Encoder, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_directions = 1
#         self.batch_size = batch_size
#         self.device = device
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
#
#     def forward(self, x):
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         output, (h, c) = self.lstm(x, (h_0, c_0))
#         return output, h, c
#
#
# class Decoder(nn.Module):
#     def __init__(self, input_length, hidden_size, num_layers=1, batch_size=1024, device=None):
#         super(Decoder, self).__init__()
#         self.input_length = input_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.input_size = hidden_size
#         self.batch_size = batch_size
#         self.device = device
#
#         self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
#                             batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, 1)
#         self.fc2 = nn.Linear(hidden_size, 1)
#         self.fc3 = nn.Linear(hidden_size, 1)
#         self.fc4 = nn.Linear(hidden_size, 1)
#         self.fc5 = nn.Linear(hidden_size, 1)
#
#     def forward(self, x, hn, cn):
#         # x = x.view(self.batch_size, 1, self.input_size)
#         lstm_out, (hn, cn) = self.lstm(x, (hn, cn))
#         pred1, pred2, pred3, pred4, pred5 = self.fc1(hn), self.fc2(hn), self.fc3(hn), self.fc4(hn), self.fc5(hn)
#         pred1, pred2, pred3, pred4, pred5 = pred1[-1, :, :], pred2[-1, :, :], pred3[-1, :, :], pred4[-1, :, :], pred5[
#                                                                                                                 -1, :,
#                                                                                                                 :]
#         out = torch.stack([pred1, pred2, pred3, pred4, pred5], dim=0)
#         return out, hn, cn
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024,
#                  device=None):
#         super(Seq2Seq, self).__init__()
#         self.input_size = input_size
#         self.input_length = input_length
#         self.output_length = output_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.device = device
#
#         self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#         self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#
#     def forward(self, x):
#         encoder_output, hidden, cell = self.encoder(x)
#         batch_length = encoder_output.shape[0]
#         encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
#         outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
#         for day_idx in range(self.output_length):
#             print(encoder_output.shape)
#             print(hidden.shape)
#             out, hidden, cell = self.decoder(encoder_output, hidden, cell)
#             outputs[:, :, day_idx] = out[:, :, -1]
#         return outputs
#
#
# class Attention(nn.Module):
#     def __init__(self, input_hid_dim, output_hid_dim):
#         super().__init__()
#         self.attention = nn.Linear(input_hid_dim + output_hid_dim, input_hid_dim)
#         self.align = nn.Linear(output_hid_dim, 1)
#
#     def forward(self, hidden, encoder_outputs):
#         batch_size = encoder_outputs.shape[0]
#         src_len = encoder_outputs.shape[1]
#         feature_size = encoder_outputs.shape[2]
#         hidden = hidden.permute(1, 0, 2)
#         hidden = hidden.repeat(1, src_len, 1)
#         attention_outputs = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
#         align_score = self.align(attention_outputs).squeeze(2)
#         attention_weights_days = F.softmax(align_score, dim=1).unsqueeze(2)
#         attention_weights_days = attention_weights_days.repeat(1, 1, feature_size)
#         attention_weight_features = F.softmax(attention_outputs, dim=2)
#         attention_weights = attention_weight_features.mul(attention_weights_days)
#         return attention_weights
#
#
# class Seq2SeqWithAttention(nn.Module):
#     def __init__(self, input_size, input_length, output_length, hidden_size, num_layers=1, batch_size=1024,
#                  device=None):
#         super(Seq2SeqWithAttention, self).__init__()
#         self.input_size = input_size
#         self.input_length = input_length
#         self.output_length = output_length
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.device = device
#
#         self.encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#         self.attention = Attention(self.input_size, self.hidden_size).to(self.device)
#         self.vector = nn.Linear(input_size + input_size, input_size)
#         self.decoder = Decoder(self.input_length, self.hidden_size, self.num_layers, self.batch_size, self.device).to(
#             self.device)
#
#     def forward(self, x):
#         encoder_output, hidden, cell = self.encoder(x)
#         batch_length = encoder_output.shape[0]
#         encoder_output = encoder_output.reshape(batch_length, self.input_length, self.hidden_size)
#         outputs = torch.zeros(5, batch_length, self.output_length).to(self.device)
#         for day_idx in range(self.output_length):
#             # print(encoder_output.shape)
#             # print(hidden.shape)
#             attention_weights = self.attention(hidden, encoder_output)
#             vector_output = self.vector(torch.cat((encoder_output, attention_weights), dim=2))
#             out, hidden, cell = self.decoder(vector_output, hidden, cell)
#             outputs[:, :, day_idx] = out[:, :, -1]
#         return outputs
